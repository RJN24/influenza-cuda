
// one kernel implementation
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <iostream>

#include <cmath>
#include <limits>



//constant values on device
__device__ __constant__ long     dev_max_number_adult;
__device__ __constant__ long     dev_max_number_child;
__device__ __constant__ long     dev_max_number_house;
__device__ __constant__ long     dev_max_number_workplaces;


//define desease transmition parameer
__device__ __constant__ float     house_trans;    //household transmition parameter
__device__ __constant__ float     place_trans;    //place transmition parameter
__device__ __constant__ float     comm_trans;     //community transmition parameter
__device__ __constant__ float     travel_rate;
__device__ __constant__ float     ind_inf;        //relative infectiousness of an individual
__device__ __constant__ int       mild_infect;    //mild infection rate w


#define BLOCK_SIZE 1024    /*size of block for division*/


#define POS_SIZE	10
#define EPSILON 0.000001



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}



// *********************************************************************************************************
// Structure ENTITY corresponds to the characteristics of a single individual of the population
struct entity
{
	// Individual attributes
    unsigned long long 	id;			// Unique # for each person in the community
    int     status;
    int		age;					// Age of the individual: 0-3 where 0 = kid, 1 = adult, 2 = senior citizen
    long	householdId;			// ID number of its associated household
    long    workPlaceId;
    int     infectedDay;
    int     x_pos;
    int     y_pos;
    int     severity;
    float   travel_rate;
    int     timer;
    bool    alive;
} entity;


struct houseHold
{
	unsigned long long 		id;							// Unique ID number
	int		type;						// Type of house, number of residents
    unsigned long long     hasInfected;            // number of infected in this place
}houseHold;

struct workPlaces  {
	unsigned long long id;					   // Unique ID number
    int                employeeNum;            // number of employee
    unsigned long long hasInfected;            // number of infected in this place
}workPlaces;

struct community {
	unsigned long long 		id;	// ID
	unsigned long long		hasInfected;	// If there is an infected currently present
}community;

// struct for day list
typedef struct list_day_node {
    struct                  list_day_node *next;
    unsigned long long      simulationDay;
    unsigned long long      numInfectedDuringDay;
    unsigned long long      totalNumInfectedAtEndOfDay;
    unsigned long long      numDeathsDuringDay;
    unsigned long long      totalDeathsAtEndOfDay;
    //int                     dayOfTheWeek;
}day_node;

struct entity *adultAgents;  /*list of all data adults on host */
struct entity *d_adultAgents;   /*list of all data adults on device */
struct entity *d_childAgents;   /*list of all data children on device */
struct list_day_node *dayUpdateList; /*list of all the daily update info*/

unsigned long long      max_number_days;
unsigned long long 		max_number_adult;
unsigned long long      max_number_households ;
unsigned long long      max_number_workplaces ;
unsigned long long      max_number_community_places = 100;
unsigned long long      max_region_population;
//long long       max_number_schools = 128;
struct houseHold *d_houseHolds;
struct workPlaces *d_workPlaces;
struct community *d_communityPlaces;
struct school *d_schools;
struct list_day_node *d_dayUpdateList;

unsigned long long  *d_infected_individuals;
unsigned long long  *infected_individuals;

__device__ unsigned long long  numberOfInfected=6 ;

const char* o_file_name = "daily-output.txt";

/* this function is used to write the changes that have happened in one day
to file. Including the number of newly infected... */
void output_to_file(FILE *myfile, struct list_day_node myday, int num_infected)
{
  // write the appropriate data from the struct to file
  fprintf(myfile, "Day %lld\n", myday.simulationDay);
  fprintf(myfile, "People infected on this day: %lld\n", myday.numInfectedDuringDay);
  fprintf(myfile, "Total number of infected: %lld\n\n", myday.totalNumInfectedAtEndOfDay+num_infected);
}



__global__ void kernel_generate_household(int startingpoint, int houseType, int residentStrttPoint , struct entity *d_adultAgents , struct houseHold *d_houseHolds, unsigned long seed) {

	const unsigned int tid = startingpoint + threadIdx.x + (blockIdx.x*blockDim.x);
    if (tid < dev_max_number_house){
        int i=0;
        d_houseHolds[tid].id=  tid;
        d_houseHolds[tid].type=houseType;
        d_houseHolds[tid].hasInfected=0;
        int aId;
        curandState_t state;
        curand_init(seed, /* the seed controls the sequence of random values that are produced */
                    tid, /* the sequence number is only important with multiple cores */
                    0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                    &state);

        int xpos = curand(&state) % POS_SIZE;
        int ypos = curand(&state) % POS_SIZE;



        int age = curand(&state) % 100;
        int ageGroup;
        if (age >= 0 && age <= 18) {    // The person is a kid  (should be an 18% chance)
            ageGroup = 0;
        }
        else if (age >= 19 && age <= 86 ) { // the person is an adult (should be a 68% chance)
            ageGroup = 1;
        }
        else {
            ageGroup = 2;   // the person is a senior citizen   (should be 14% chance)
        }

        // ensure we arent going over max number of adults
        for (i=0; i<houseType && tid+i+tid*(houseType-1) < dev_max_number_adult; i++){

            aId= tid + i +  tid*(houseType-1);

            d_adultAgents[aId].id=aId;

            d_adultAgents[aId].status=0;
            d_adultAgents[aId].householdId=tid;

            d_adultAgents[aId].age=ageGroup;
            d_adultAgents[aId].infectedDay=0;
            d_adultAgents[aId].severity=0;
            d_adultAgents[aId].alive=true;
            d_adultAgents[aId].x_pos =xpos;
            d_adultAgents[aId].y_pos =ypos;
            d_adultAgents[aId].travel_rate = 1.0;
        }

    }

    __syncthreads();

}

__global__ void kernel_generate_workplace(int numberofEmployee, struct entity *d_adultAgents, struct workPlaces *d_workPlaces) {

    const unsigned int tid = threadIdx.x + (blockIdx.x*blockDim.x);
    int i;
    if( tid < dev_max_number_workplaces ){
        d_workPlaces[tid].id= tid ;
        d_workPlaces[tid].employeeNum= numberofEmployee;
        d_workPlaces[tid].hasInfected=0;
        int id=0;

        for (i=0; i<numberofEmployee && tid+i+(tid*(numberofEmployee-1)) < dev_max_number_adult; i++){
            id = tid +  i + ( tid * (numberofEmployee-1));
            d_adultAgents[id].workPlaceId = tid;
        }
    }

    __syncthreads();

}

__global__ void kernel_generate_community(struct community *d_community) {

    const unsigned int tid = threadIdx.x + (blockIdx.x*blockDim.x);
    //int i;

    d_community[tid].id=tid;
    //d_community[tid].type = communityType; // not needed
    d_community[tid].hasInfected = 0;
    __syncthreads();
}


// initializes the original 6 infected people.
__global__ void kernel_update_infected( unsigned long long  id0, unsigned long long  id1,  unsigned long long  id2,  unsigned long long  id3,  unsigned long long  id4,  unsigned long long  id5, struct entity *d_adultAgents){
    d_adultAgents[id0].severity=1;
    d_adultAgents[id0].status=1;
    d_adultAgents[id0].infectedDay=0;
    d_adultAgents[id0].timer=0;
    d_adultAgents[id0].alive=true;
    d_adultAgents[id1].severity=1;
    d_adultAgents[id1].status=1;
    d_adultAgents[id1].infectedDay=0;
    d_adultAgents[id1].timer=0;
    d_adultAgents[id1].alive=true;
    d_adultAgents[id2].severity=1;
    d_adultAgents[id2].status=1;
    d_adultAgents[id2].infectedDay=0;
    d_adultAgents[id2].timer=0;
    d_adultAgents[id2].alive=true;
    d_adultAgents[id3].severity=1;
    d_adultAgents[id3].status=1;
    d_adultAgents[id3].infectedDay=0;
    d_adultAgents[id3].timer=0;
    d_adultAgents[id3].alive=true;
    d_adultAgents[id4].severity=1;
    d_adultAgents[id4].status=1;
    d_adultAgents[id4].infectedDay=0;
    d_adultAgents[id4].timer=0;
    d_adultAgents[id4].alive=true;
    d_adultAgents[id5].severity=1;
    d_adultAgents[id5].status=1;
    d_adultAgents[id5].infectedDay=0;
    d_adultAgents[id5].timer=0;
    d_adultAgents[id5].alive=true;

}

__device__ float inline calculatePointDistance(int x1, int y1, int x2, int y2, int a, float b)
{
	float sqx = (x1-x2) * (x1-x2);
	float sqy = (y1-y2) * (y1-y2);
	//return ( 1/ pow( (1 + (sqrt(sqx + sqy ) / a )) ,b ));
    return sqrt(sqx + sqy );
}


__global__ void kernel_calculate_contact_process(  unsigned long long  *d_infected_individuals,
    struct entity *d_adultAgents, struct houseHold *d_houseHolds, struct workPlaces *d_workPlaces,
    struct community *d_communityPlaces, unsigned long long max_community_places, int simulationDay,
    struct list_day_node* daily_list, unsigned int seed ) {

    const unsigned long long tid =  threadIdx.x + (blockIdx.x*blockDim.x);

    int weekDayStatus;
    int currentHour;
    double check_rand;
    register double cur_lambda=0.0;
    int homeHours, workHours, comHours;
    homeHours = 0;
    workHours = 0;
    comHours = 0;

    if( tid < dev_max_number_adult ){


        unsigned long hid = d_adultAgents[tid].householdId;
        unsigned long wid = d_adultAgents[tid].workPlaceId;
        unsigned long communityId=0;

        curandState_t state;
        curand_init(seed, /* the seed controls the sequence of random values that are produced */
                    tid, /* the sequence number is only important with multiple cores */
                    simulationDay, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                    &state);

        // weekdays are satuday and sunday (5 and 6) since we include day 0
        if (simulationDay%6==0 || (simulationDay+1)%6==0){
            weekDayStatus=0;
        }
        else {
            weekDayStatus=1;
        }
        cur_lambda=0;

        // clear out infections at each workplace/household/communityplace
        d_workPlaces[wid].hasInfected = 0;
        d_houseHolds[hid].hasInfected = 0;
        if( tid < max_community_places ){
            d_communityPlaces[tid].hasInfected = 0;
        }



        // now set the daily variables to 0
        if( tid == 0 ){
            daily_list[simulationDay].numInfectedDuringDay = 0;
            if(simulationDay == 0){
                daily_list[simulationDay].totalNumInfectedAtEndOfDay = 0;
            }
            else {
                daily_list[simulationDay].totalNumInfectedAtEndOfDay =
                    daily_list[simulationDay-1].totalNumInfectedAtEndOfDay;
            }
        }



        // loop through the day hour by hour if alive
        if( d_adultAgents[tid].alive == true ){
            for (currentHour = 0; currentHour < 24; currentHour ++){

                //adults are at home on weekday or weekends
                if  ( ( (weekDayStatus == 1) && (( currentHour>= 19 &&   currentHour<24) || ( currentHour>= 0 &&   currentHour< 8)) ) ||  ( (weekDayStatus == 0) && (( currentHour>= 0 &&   currentHour<17) || ( currentHour>= 19 &&   currentHour< 24)) ) ){

                    // check if anyone in the house is sick
                    if( d_adultAgents[tid].status == 1 && homeHours == 0 ){
                        atomicAdd(&d_houseHolds[hid].hasInfected, 1); // add one infected there
                    }
                    homeHours += 1;

                    // OLD LAMDA code
                    // for (j = 0; j < numberOfInfected; j++){
                    //     //call the function for d_adultAgents[tid].id va infectedid
                    //     if (d_adultAgents[d_infected_individuals[j]].householdId == houseid){
                    //         cur_lambda = cur_lambda + ((house_trans * ( 0.1255 * exp(- ( pow ((log((double) ( (simulationDay-d_adultAgents[d_infected_individuals[j]].infectedDay) + 0.72) )), 2.0) / 6.48) ) )* (1 + d_adultAgents[d_infected_individuals[j]].severity) ) / (pow((double)d_houseHolds[houseid].type, 0.8)));
                    //     }
                    // }
                }
                //adults are at work or school 8am-5pm
                else if ( (weekDayStatus == 1) && ( currentHour >= 8 && currentHour < 17 ) ){

                    // check if this is an infected thread, if so set the workplace to hasInfected
                    if( d_adultAgents[tid].status == 1 && workHours == 0 ){
                        atomicAdd(&d_workPlaces[wid].hasInfected,1); // add one infected there
                    }
                    workHours += 1;

                    // OLD LAMDA CODE
                    // for (int j = 0; j < numberOfInfected; j++){
                    //     if (d_adultAgents[d_infected_individuals[j]].workPlaceId == workplaceId){
                    //         //call the function for d_adultAgents[tid].id va infectedid
                    //         if ( simulationDay - d_adultAgents[d_infected_individuals[j]].infectedDay > 0.25) {
                    //             cur_lambda = cur_lambda + ( (place_trans * ( 0.1255 * exp(- ( pow ((log((double) ( (simulationDay-d_adultAgents[d_infected_individuals[j]].infectedDay) + 0.72) )), 2.0) / 6.48) ) ) * (1 + d_adultAgents[d_infected_individuals[j]].severity * ( (2 * 0.5) -1)) ) / d_workPlaces [workplaceId].employeeNum ) ;
                    //         }
                    //     }
                    // }
                }
                //adults on weekdays errand -- community events 5-7pm
                else if ( currentHour >= 17 && currentHour < 19  ){
                    // assign each thread to a random community event
                    check_rand = curand(&state) % max_community_places;
                    communityId = check_rand; // this thread will be at this community

                    // check if this is an infected thread, if so set the communityPlace to infected
                    if (d_adultAgents[tid].status == 1 && comHours == 0 ) {
                        atomicAdd(&d_communityPlaces[communityId].hasInfected,1);
                    }
                    comHours += 1;

                    // OLD LAMDA CODE
                    // float current_distance =0 ;
                    // for (int j = 0; j < numberOfInfected; j++){
                    //     current_distance = calculatePointDistance ( d_adultAgents[tid].x_pos, d_adultAgents[tid].y_pos, d_adultAgents[d_infected_individuals[j]].x_pos, d_adultAgents[d_infected_individuals[j]].y_pos, 35, 6.5 );
                    //     if ( fabs(current_distance - 2.000000) < EPSILON ){
                    //         cur_lambda = cur_lambda +  (1 * comm_trans * ( 0.1255 * exp(- ( pow ((log((double) ( (simulationDay-d_adultAgents[d_infected_individuals[j]].infectedDay) + 0.72) )), 2.0) / 6.48) ) ) * current_distance * (1 + d_adultAgents[d_infected_individuals[j]].severity)) ;
                    //     }
                    // }
                }
            } // end for loop
        } // end if - alive

        // only applies to people who are not infected and alive
        if (d_adultAgents[tid].status==0 && d_adultAgents[tid].alive == true) {

            // new lamda code for homes
            if( d_houseHolds[hid].hasInfected >= 1){
                // there is an infected person in the house, take this into accnt for each infected person
                for( int x=0; x<d_houseHolds[hid].hasInfected; x++){
                    // calc our log_normal_distribution
                    check_rand = curand_log_normal(&state,-0.72,1.8);
                    // using equation from parameters file in google drive
                    cur_lambda = cur_lambda + ( (0.01958*homeHours*check_rand*2) / d_houseHolds[hid].type);
                    //cur_lambda = cur_lambda + ((house_trans * ( 0.1255 * exp(- ( pow ((log((double) (1.72) )), 2.0) / 6.48) ) )* (2) ) / (pow((double)d_houseHolds[hid].type, 0.8)));
                }
            }

            // new lamda code for work places
            if(d_workPlaces[wid].hasInfected >= 1){
                for( int x=0; x<d_workPlaces[wid].hasInfected; x++){
                    check_rand = curand_log_normal(&state,-0.72,1.8);
                    //printf("thread %lld has distribution %f\n", tid, check_rand);
                    // using equation from parameters file in google drive
                    cur_lambda = cur_lambda + ((0.01958*workHours*check_rand) / d_workPlaces[wid].employeeNum);
                    //cur_lambda = cur_lambda + ( (place_trans * ( 0.1255 * exp(- ( pow ((log((double) ( 1.72) )), 2.0) / 6.48) ) ) * (( (2 * 0.5) -1)) ) / d_workPlaces [wid].employeeNum ) ;
                }
            }

            // new lamda code for community places
            if( d_communityPlaces[communityId].hasInfected >= 1){
                for( int x=0; x<d_communityPlaces[communityId].hasInfected; x++){
                    check_rand = curand_log_normal(&state,-0.72, 1.8);
                    // using equation from parameters file in google drive
                    cur_lambda = cur_lambda + ((0.003125*comHours*check_rand*2));
                    //cur_lambda = cur_lambda +  (1 * comm_trans * ( 0.1255 * exp(- ( pow ((log((double)(1.72) )), 2.0) / 6.48) ) ) * 2) ;
                }
            }

            //printf("%llu, current lamda before negative exp: %f\n",tid,cur_lambda);
            cur_lambda = (1 - exp((-1)*cur_lambda)) ;
            check_rand = curand_uniform(&state);
            //printf("%llu, Current lamda: %f, Current dist: %f\n",tid,cur_lambda,check_rand);
            //bool check =fabs(cur_lambda - check_rand) < EPSILON;

            // per project description, cur_lambda now hold the probability of this person
            // becoming infected.
            // thus, if the rand distribution is less, then they are infected
            bool check = (check_rand < cur_lambda);
            if (check==true) {
                d_adultAgents[tid].status=1;
                d_adultAgents[tid].infectedDay= simulationDay;
                d_adultAgents[tid].severity = 1;
                d_adultAgents[tid].timer = 0;

                unsigned long long  my_idx = atomicAdd(&numberOfInfected, 1);
                d_infected_individuals[my_idx] = d_adultAgents[tid].id;
                atomicAdd(&daily_list[simulationDay].numInfectedDuringDay,1);
            }
        // if a person is infected and still alive, increment its timer
        // if it has been 7 days, the person has a 38.6% chance to die
        // status is set to 0 whether they live or die
        }
        else if (d_adultAgents[tid].status == 1 && d_adultAgents[tid].alive) {
            d_adultAgents[tid].timer++;
            if (d_adultAgents[tid].timer == 7) {
                d_adultAgents[tid].status = 0;
                check_rand = curand_uniform(&state);
                //double result = check_rand % 100;//curand_uniform_double(&state) % 100;
                if(check_rand < 0.3861){//if (result < 38.61) {
                    d_adultAgents[tid].alive = false;
                }
            }
        }

    }// end if tid < dev_max_number_adult

    __syncthreads();

    // now set the total number of infected
    if( simulationDay > 0 && tid == 0){ // run this only once (hence tid==0), on all days after day 1
        // unsigned long long tempTotal = daily_list[simulationDay-1].totalNumInfectedAtEndOfDay +
        //     daily_list[simulationDay].numInfectedDuringDay;
        atomicAdd(&daily_list[simulationDay].totalNumInfectedAtEndOfDay,daily_list[simulationDay].numInfectedDuringDay);
        // set the day number in the daily output struct
        daily_list[simulationDay].simulationDay = simulationDay;
    }
    else if(tid == 0){ // run this only once per loop (hence tid==0), on day one
        daily_list[simulationDay].totalNumInfectedAtEndOfDay = daily_list[simulationDay].numInfectedDuringDay;
        // set the day number in the daily output struct
        daily_list[simulationDay].simulationDay = simulationDay;
    }
    __syncthreads();
}




int main(int argc, const char * argv[])
{
    if( argc < 3 )
    {
      printf("Usage: ./a.out <number of adults> <number of days>\n");
      return 1;
    }

    max_number_adult = atoi(argv[1]);
    if( max_number_adult < 6 )
        max_number_adult = 6;

    max_number_days = atoi(argv[2]);
    if( max_number_days <= 0)
        max_number_days = 1;
    max_number_households = max_number_adult/5;

    int max_num_employee= 100;
    max_number_workplaces= (max_number_adult / max_num_employee) + 1;

    //unsigned long long  i;
    unsigned long long  j;
    unsigned long long h_numberOfInfected=6;
    int num_infected=6;
    unsigned int s; // used for seeding random

    //printf( "start allocation \n" );

    adultAgents = (struct entity *)malloc(sizeof(struct entity)*max_number_adult);
    memset(adultAgents, 0,  (sizeof(struct entity)*max_number_adult) );


    infected_individuals = (unsigned long long  *) malloc(sizeof(unsigned long long )*max_number_adult);
    memset(infected_individuals, 0,  (sizeof(unsigned long long )*max_number_adult) );

    // set up our day update list
    dayUpdateList = (struct list_day_node*)malloc(sizeof(struct list_day_node)*max_number_days);

    //printf( "start allocation on device \n" );

    cudaMalloc((void **) &d_adultAgents, sizeof(struct entity)*max_number_adult );

    cudaMalloc((void **) &d_houseHolds, sizeof(struct houseHold)*max_number_households );
    cudaMalloc((void **) &d_workPlaces, sizeof(struct workPlaces ) * max_number_workplaces);
    cudaMalloc((void **) &d_communityPlaces, sizeof(struct community)*max_number_community_places);

    cudaMalloc((void **) &d_infected_individuals, sizeof(unsigned long long) * ( max_number_adult));

    // allocate the dayUpdateList on GPU
    cudaMalloc((void **) &d_dayUpdateList, sizeof(struct list_day_node)*max_number_days);

    //printf( "finish allocation \n" );

    cudaMemset(d_adultAgents, 0, sizeof(struct entity)*max_number_adult);
    //  cudaMemset(d_childAgents, 0, sizeof(struct entity)*max_number_child);
    cudaMemset(d_houseHolds, 0, sizeof(struct houseHold)*max_number_households);
    cudaMemset(d_workPlaces, 0, sizeof(struct workPlaces)*max_number_workplaces);
    cudaMemset(d_communityPlaces, 0, sizeof(struct community)*max_number_community_places);
    cudaMemset(d_infected_individuals, 0, sizeof(unsigned long long )* ( max_number_adult));
    cudaMemset(d_dayUpdateList,0,sizeof(struct list_day_node)*max_number_days);

    cudaMemcpyToSymbol(dev_max_number_adult, &max_number_adult, sizeof(unsigned long long ));
    // cudaMemcpyToSymbol(dev_max_number_child, &max_number_child, sizeof(unsigned long long ));
    cudaMemcpyToSymbol(dev_max_number_house, &max_number_households, sizeof(unsigned long long ));
    cudaMemcpyToSymbol(dev_max_number_workplaces, &max_number_workplaces, sizeof(unsigned long long));
    float house_transmition = 0.47;
    float place_transmition = 0.94;
    float community_transmition = 0.075;
    cudaMemcpyToSymbol(house_trans, &house_transmition , sizeof(float));
    cudaMemcpyToSymbol(place_trans, &place_transmition, sizeof(float));
    cudaMemcpyToSymbol(comm_trans, &community_transmition, sizeof(float));

    for(j = 0;  j < num_infected; j++) {
        infected_individuals[j]=rand() % max_number_adult ;
        //printf( "infected_individuals id is = %llu : \n", infected_individuals[j]);
    }

    int blocks_num;
    int startpoint=0;

    int houseType = 5;
    int residentStrttPoint=0;

    //blocks_num = (max_number_adult) / BLOCK_SIZE;
    blocks_num = ceil((max_number_adult) / BLOCK_SIZE)+1;

    dim3 grid(blocks_num, 1, 1);
    dim3 threads(BLOCK_SIZE, 1, 1);
    //printf( "number of blocks is: %i \n", blocks_num );

    /* start counting time */
    cudaEvent_t start, stop;
    //    cudaEventCreate(&start);
    //    cudaEventCreate(&stop);
    //    cudaEventRecord(start, 0);
    //
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    cudaMemcpy(d_infected_individuals, infected_individuals, sizeof(unsigned long long )*max_number_adult, cudaMemcpyHostToDevice);

    gpuErrchk(cudaDeviceSynchronize());
    //call the kernel function to generate individual of household type 1 , 2 adults, 3 children

    //printf( "******** before calling the function ***** \n");
    s = time(0);
    kernel_generate_household <<<grid, threads>>> ( startpoint,  houseType, residentStrttPoint, d_adultAgents ,d_houseHolds, s);


    //printf( "******** after calling the function ***** \n");
    gpuErrchk(cudaDeviceSynchronize());


    blocks_num = ceil(max_number_workplaces / BLOCK_SIZE)+1;
    //printf( "number of blocks for work places is: %i \n", blocks_num );
    dim3 grid2(blocks_num, 1, 1);
    dim3 threads2(BLOCK_SIZE, 1, 1);
    kernel_generate_workplace << <grid2, threads2>> > ( max_num_employee, d_adultAgents, d_workPlaces);
    //move infected individuals

    gpuErrchk(cudaDeviceSynchronize());

    blocks_num = ceil(max_number_community_places / BLOCK_SIZE)+1;
    //printf( "number of blocks for community places is: %i\n", blocks_num );
    dim3 gridCom(blocks_num,1,1);
    dim3 threadsCom(BLOCK_SIZE,1,1);
    kernel_generate_community <<< gridCom, threadsCom >>> (d_communityPlaces);

    gpuErrchk(cudaDeviceSynchronize());

    //update infected individuals status
    dim3 grid3(1, 1, 1);
	dim3 threads3(1, 1, 1);
    kernel_update_infected <<<grid3, threads3>>> (infected_individuals[0],
        infected_individuals[1], infected_individuals[2], infected_individuals[3],
        infected_individuals[4], infected_individuals[5], d_adultAgents);
    gpuErrchk(cudaDeviceSynchronize());

    blocks_num = ceil((max_number_adult) / BLOCK_SIZE)+1;
    dim3 grid4(blocks_num, 1, 1);
    dim3 threads4(BLOCK_SIZE, 1, 1);


    printf("Starting simulation for %d people over %d days.\n", max_number_adult, max_number_days);
    //run the simulation
    int simulationDay;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //printf( "******** call the contact process ***** \n");
    //calclate force of infection
    //should be run for each day of simulation
    for (simulationDay=0; simulationDay < max_number_days; simulationDay++){
        s = time(0);
        kernel_calculate_contact_process <<<grid4, threads4>>> ( d_infected_individuals,
            d_adultAgents, d_houseHolds, d_workPlaces, d_communityPlaces, max_number_community_places, simulationDay,
            d_dayUpdateList, s);
    }

    gpuErrchk(cudaDeviceSynchronize());


    /* check the total running time */
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    printf( "******** Total Running Time of Kernel = %0.5f ms ********\n", elapsedTime );

    //copy adults back to cpu
    //cudaMemcpy(adultAgents, d_adultAgents, sizeof(struct entity)*max_number_adult, cudaMemcpyDeviceToHost);
    cudaMemcpy(infected_individuals, d_infected_individuals, sizeof(unsigned long long )*max_number_adult, cudaMemcpyDeviceToHost);

    cudaMemcpyFromSymbol(&h_numberOfInfected, numberOfInfected, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&h_numberOfInfected, numberOfInfected, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // copy the day list back to cpu
    cudaMemcpy(dayUpdateList, d_dayUpdateList, sizeof(struct list_day_node)*max_number_days, cudaMemcpyDeviceToHost);

    // set up our output FILE
    // file opening and error checking
    FILE *myfile = fopen(o_file_name, "w+"); // erases/creates the file
    if( !myfile ){
      printf("Error opening output file.\n");
    }
    else{
        fprintf(myfile, "Starting day by day output of simulation!\n");
        fprintf(myfile, "Using the following parameters: \n");
        fprintf(myfile, "Max number of agents: %lld\n", max_number_adult);
        fprintf(myfile, "Number of simulated days: %d\n", max_number_days);
        fprintf(myfile, "Starting with %d initial infected.\n\n", num_infected);
        for(int i=0; i<simulationDay; i++){
            output_to_file(myfile,dayUpdateList[i], num_infected);
        }
    }
    fclose(myfile);
    printf("Check daily-output.txt for a day-by-day log of the simulation!\n");

    /* Clean up memory */

    cudaFree(d_adultAgents);
    cudaFree(d_houseHolds);
    cudaFree(d_workPlaces);
    cudaFree(d_communityPlaces);
    cudaFree(d_infected_individuals);


    free(adultAgents);
    free(infected_individuals);


    return 0;
}
