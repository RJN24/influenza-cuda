
// one kernel implementation
#include <unistd.h>
#include <stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include<time.h>
#include<string.h>
#include <iostream>

#include <cmath>
#include <limits>



//constant values on device
__device__ __constant__ long     dev_max_number_adult;
__device__ __constant__ long     dev_max_number_child;
__device__ __constant__ long     dev_max_number_house;


//define desease transmition parameer
__device__ __constant__ float     house_trans;    //household transmition parameter
__device__ __constant__ float     place_trans;    //place transmition parameter
__device__ __constant__ float     comm_trans;      //community transmition parameter
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
    int		age;					// Age of the individual: 0-7 basedon age group in input data
    long	householdId;			// ID number of its associated household
    long    workPlaceId;
    int     infectedDay;
    int     x_pos;
    int     y_pos;
    int     severity;
    float   travel_rate;
} entity;


struct houseHold
{
	unsigned long long 		id;							// Unique ID number
	int		type;						// Type of house, number of residents
    int     hasInfected;            //if there is an infected in this place
}houseHold;

struct workPlaces  {
	unsigned long long 		id;							// Unique ID number
    int         employeeNum;                //number of employee
    int         hasInfected;            //if there is an infected in this place
}workPlaces;

// this struct may be unnecessary
struct community {
	unsigned long long 		id;	// ID
	int 		type;		// type of community
	int 		hasInfected;	// If there is an infected currently present
}community;

// struct for day list
typedef struct list_day_node {
    struct                  list_day_node *next;
    unsigned long long      simulationDay;
    unsigned long long      numInfectedDuringDay;
    unsigned long long      totalNumInfectedAtEndOfDay;
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
unsigned long long      max_region_population;
//long long       max_number_schools = 128;
struct houseHold *d_houseHolds;
struct workPlaces *d_workPlaces;
struct school *d_schools;
struct list_day_node *d_dayUpdateList;

unsigned long long  *d_infected_individuals;
unsigned long long  *infected_individuals;

__device__ unsigned long long  numberOfInfected=9 ;

const string o_file_name = "daily-output.txt";

/* this function is used to write the changes that have happened in one day
to file. Including the number of newly infected... */
void output_to_file(FILE myfile, struct list_day_node myday)
{
  // write the appropriate data from the struct to file
  fprintf(myfile, "Day %lld\n", myday.simulationDay);
  fprintf(myfile, "People infected on this day: %lld\n", myday.numInfectedDuringDay);
  fprintf(myfile, "Total number of infected on this day: %lld\n\n", myday.totalNumInfectedAtEndOfDay);
}



__global__ void kernel_generate_household(int startingpoint, int houseType, int residentStrttPoint , struct entity *d_adultAgents , struct houseHold *d_houseHolds) {



	const unsigned int tid = startingpoint + threadIdx.x + (blockIdx.x*blockDim.x);
    if (tid < dev_max_number_house){
        int i=0;
        d_houseHolds[tid].id=  tid;
        d_houseHolds[tid].type=houseType;
        d_houseHolds[tid].hasInfected=0;
        int aId;
        //curandState_t state = states[tid];
        curandState_t state;
        curand_init(0, /* the seed controls the sequence of random values that are produced */
                    tid, /* the sequence number is only important with multiple cores */
                    0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                    &state);

        int xpos = curand(&state) % POS_SIZE;
        int ypos = curand(&state) % POS_SIZE;

//        if(tid==10) {
//            printf("xpos is: %i  and ypos is %i : \n", xpos, ypos );
//        }

        for (i=0; i<houseType; i++){

            aId= tid + i +  tid*(houseType-1);

            d_adultAgents[aId].id=aId;

            d_adultAgents[aId].status=0;
            d_adultAgents[aId].householdId=tid;

            d_adultAgents[aId].age=23;
            d_adultAgents[aId].infectedDay=0;
            d_adultAgents[aId].severity=0;
            d_adultAgents[aId].x_pos =xpos;
            d_adultAgents[aId].y_pos =ypos;
            d_adultAgents[aId].travel_rate = 1.0;
            // todo add travel rate based on age
            // NOTE - the TA said this is not needed for our implementation
        }



//        if(tid==10) {
//            printf("household id is: %li and adult agent id is %li  \n", d_houseHolds[tid].id, d_adultAgents[aId].id);
//        }

    }

    __syncthreads();

}

__global__ void kernel_generate_workplace(int numberofEmployee, struct entity *d_adultAgents, struct workPlaces *d_workPlaces) {

    const unsigned int tid = threadIdx.x + (blockIdx.x*blockDim.x);
    int i;

    d_workPlaces[tid].id= tid ;
    d_workPlaces[tid].employeeNum= numberofEmployee;
    d_workPlaces[tid].hasInfected=0;
    int id=0;

    for (i=0; i<numberofEmployee; i++){
        id = tid +  i + ( tid * (numberofEmployee-1));
        d_adultAgents[id].workPlaceId = tid;

    }
    if(tid==1) {
        printf("here\n");
        printf("adult workplace id is:%li and agent id is: %i  \n", d_adultAgents[id].id, tid);

    }

    __syncthreads();

}




// initializes the original 9 infected people.
// this is only done on one thread...
__global__ void kernel_update_infected( unsigned long long  id0, unsigned long long  id1,  unsigned long long  id2,  unsigned long long  id3,  unsigned long long  id4,  unsigned long long  id5,  unsigned long long  id6,  unsigned long long  id7,  unsigned long long  id8,  unsigned long long  id9, struct entity *d_adultAgents){

    d_adultAgents[id0].severity=1;
    d_adultAgents[id0].status=1;
    d_adultAgents[id0].infectedDay=0;
    d_adultAgents[id1].severity=0;
    d_adultAgents[id1].status=1;
    d_adultAgents[id1].infectedDay=0;
    d_adultAgents[id2].severity=1;
    d_adultAgents[id2].status=1;
    d_adultAgents[id2].infectedDay=0;
    d_adultAgents[id3].severity=0;
    d_adultAgents[id3].status=1;
    d_adultAgents[id3].infectedDay=0;
    d_adultAgents[id4].severity=0;
    d_adultAgents[id4].status=1;
    d_adultAgents[id4].infectedDay=0;
    d_adultAgents[id5].severity=1;
    d_adultAgents[id5].status=1;
    d_adultAgents[id5].infectedDay=0;
    d_adultAgents[id6].severity=1;
    d_adultAgents[id6].status=1;
    d_adultAgents[id6].infectedDay=0;
    d_adultAgents[id7].severity=1;
    d_adultAgents[id7].status=1;
    d_adultAgents[id7].infectedDay=0;
    d_adultAgents[id8].severity=0;
    d_adultAgents[id8].status=1;
    d_adultAgents[id8].infectedDay=0;
    d_adultAgents[id9].severity=1;
    d_adultAgents[id9].status=1;
    d_adultAgents[id9].infectedDay=0;


}

__device__ float inline calculatePointDistance(int x1, int y1, int x2, int y2, int a, float b)
{
	float sqx = (x1-x2) * (x1-x2);
	float sqy = (y1-y2) * (y1-y2);
	//return ( 1/ pow( (1 + (sqrt(sqx + sqy ) / a )) ,b ));
    return sqrt(sqx + sqy );
}


__global__ void kernel_calculate_contact_process(  unsigned long long  *d_infected_individuals,
    struct entity *d_adultAgents, struct houseHold *d_houseHolds,
    struct workPlaces *d_workPlaces, int simulationDay, unsigned long long max_n ) {

    const unsigned int tid =  threadIdx.x + (blockIdx.x*blockDim.x);

    int weekDayStatus;
    int currentHour;
    float check_rand;
    register double cur_lambda=0;

    // if we are on day 1 set the age of our agent
    // if( simulationDay == 1 ){
    //     // set the age of the agent based on the proportions given in the input data
    //     if( tid / max_n < 0.06 ){
    //         d_adultAgents[tid].age = 0;
    //         d_adultAgents[tid].travel_rate = 0.0;
    //     }
    //     else if( tid / max_n < 0.12 ){
    //         d_adultAgents[tid].age = 1;
    //         d_adultAgents[tid].travel_rate = 0.25;
    //     }
    //     else if( tid / max_n < 0.18 ){
    //         d_adultAgents[tid].age = 2;
    //         d_adultAgents[tid].travel_rate = 0.50;
    //     }
    //     else if( tid / max_n < 0.21 ){
    //         d_adultAgents[tid].age = 3;
    //         d_adultAgents[tid].travel_rate = 0.75;
    //     }
    //     else if( tid / max_n < 0.31 ){
    //         d_adultAgents[tid].age = 4;
    //         d_adultAgents[tid].travel_rate = 0.75;
    //     }
    //     else if( tid / max_n < 0.42 ){
    //         d_adultAgents[tid].age = 5;
    //         d_adultAgents[tid].travel_rate = 1.0;
    //     }
    //     else if( tid / max_n < 0.86 ){
    //         d_adultAgents[tid].age = 6;
    //         d_adultAgents[tid].travel_rate = 1.0;
    //     }
    //     // set remaining to > 65 years old
    //     else{
    //         d_adultAgents[tid].age = 7;
    //         d_adultAgents[tid].travel_rate = 0.75;
    //     }
    // }
    if (simulationDay%7==0 || simulationDay%7==0){
        weekDayStatus=0;
    }
    else {
        weekDayStatus=1;
    }
    cur_lambda=0;


    for (currentHour =0; currentHour <24; currentHour ++){

        if (d_adultAgents[tid].status==0) {

           // float total_distance =0;
            int j;


            //adults are at home on weekday or weekends
            if  ( ( (weekDayStatus == 1) && (( currentHour>= 19 &&   currentHour<24) || ( currentHour>= 0 &&   currentHour< 8)) ) ||  ( (weekDayStatus == 0) && (( currentHour>= 0 &&   currentHour<17) || ( currentHour>= 19 &&   currentHour< 24)) ) ){
                unsigned long long  houseid =d_adultAgents[tid].householdId;

                for (j = 0; j < numberOfInfected; j++){
                    //call the function for d_adultAgents[tid].id va infectedid
                    if (d_adultAgents[d_infected_individuals[j]].householdId == houseid){

                        cur_lambda = cur_lambda + ((house_trans * ( 0.1255 * exp(- ( pow ((log((double) ( (simulationDay-d_adultAgents[d_infected_individuals[j]].infectedDay) + 0.72) )), 2.0) / 6.48) ) )* (1 + d_adultAgents[d_infected_individuals[j]].severity) ) / (pow((double)d_houseHolds[houseid].type, 0.8)));


                    }
                }


            }


        }

        //adults are at work or school 8am-5pm
        else if ( (weekDayStatus == 1) && ( currentHour>= 8 &&   currentHour<17 ) ){

            unsigned long long  workplaceId =d_adultAgents[tid].workPlaceId;

            for (j = 0; j < numberOfInfected; j++){
                if (d_adultAgents[d_infected_individuals[j]].workPlaceId == workplaceId){
                    //call the function for d_adultAgents[tid].id va infectedid
                    if ( simulationDay - d_adultAgents[d_infected_individuals[j]].infectedDay > 0.25) {
                        cur_lambda = cur_lambda + ( (place_trans * ( 0.1255 * exp(- ( pow ((log((double) ( (simulationDay-d_adultAgents[d_infected_individuals[j]].infectedDay) + 0.72) )), 2.0) / 6.48) ) ) * (1 + d_adultAgents[d_infected_individuals[j]].severity * ( (2 * 0.5) -1)) ) / d_workPlaces [workplaceId].employeeNum ) ;

            //adults are at work or school
            else if ( (weekDayStatus == 1) && ( currentHour>= 8 &&   currentHour<17 ) ){

                unsigned long long  workplaceId =d_adultAgents[tid].workPlaceId;

                for (j = 0; j < numberOfInfected; j++){
                    if (d_adultAgents[d_infected_individuals[j]].workPlaceId == workplaceId){
                        //call the function for d_adultAgents[tid].id va infectedid
                        if ( simulationDay - d_adultAgents[d_infected_individuals[j]].infectedDay > 0.25) {
                            cur_lambda = cur_lambda + ( (place_trans * ( 0.1255 * exp(- ( pow ((log((double) ( (simulationDay-d_adultAgents[d_infected_individuals[j]].infectedDay) + 0.72) )), 2.0) / 6.48) ) ) * (1 + d_adultAgents[d_infected_individuals[j]].severity * ( (2 * 0.5) -1)) ) / d_workPlaces [workplaceId].employeeNum ) ;
                        }
                    }
                }

            }



        }

        //adults on weekdays errand -- community events 5-7pm
        else if ( currentHour>= 17 &&   currentHour<19  ){


            float current_distance =0 ;

            for (j = 0; j < numberOfInfected; j++){

                current_distance = calculatePointDistance ( d_adultAgents[tid].x_pos, d_adultAgents[tid].y_pos, d_adultAgents[d_infected_individuals[j]].x_pos, d_adultAgents[d_infected_individuals[j]].y_pos, 35, 6.5 );
                if ( fabs(current_distance - 2.000000) < EPSILON ){
                    cur_lambda = cur_lambda +  (1 * comm_trans * ( 0.1255 * exp(- ( pow ((log((double) ( (simulationDay-d_adultAgents[d_infected_individuals[j]].infectedDay) + 0.72) )), 2.0) / 6.48) ) ) * current_distance * (1 + d_adultAgents[d_infected_individuals[j]].severity)) ;

            //adults on weekdays errand
            else if ( currentHour>= 17 &&   currentHour<19  ){


                float current_distance =0 ;

                for (j = 0; j < numberOfInfected; j++){

                    current_distance = calculatePointDistance ( d_adultAgents[tid].x_pos, d_adultAgents[tid].y_pos, d_adultAgents[d_infected_individuals[j]].x_pos, d_adultAgents[d_infected_individuals[j]].y_pos, 35, 6.5 );
                    if ( fabs(current_distance - 2.000000) < EPSILON ){
                        cur_lambda = cur_lambda +  (1 * comm_trans * ( 0.1255 * exp(- ( pow ((log((double) ( (simulationDay-d_adultAgents[d_infected_individuals[j]].infectedDay) + 0.72) )), 2.0) / 6.48) ) ) * current_distance * (1 + d_adultAgents[d_infected_individuals[j]].severity)) ;
                    }

                }



            }



        }
    }
    // here is the alpha variable.
    //float alpha = 0.8;


    if (d_adultAgents[tid].status==0) {


        cur_lambda = (1- exp (- cur_lambda)) ;

        curandState_t state;
        curand_init(0, /* the seed controls the sequence of random values that are produced */
                    tid, /* the sequence number is only important with multiple cores */
                    0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                    &state);
        check_rand = curand_uniform(&state) ;

        bool check =fabs(cur_lambda - check_rand) < EPSILON;
        if (check==true) {
            d_adultAgents[tid].status=1;
            d_adultAgents[tid].infectedDay= simulationDay;
            d_adultAgents[tid].severity = 1;
             //d_adultAgents[tid].severity = curand(&state) % 2;

            unsigned long long  my_idx = atomicAdd(&numberOfInfected, 1);
            d_infected_individuals[my_idx] = d_adultAgents[tid].id;


        }



    }
__syncthreads();


}




int main(int argc, const char * argv[])
{
    if( argc < 3 )
    {
      printf("Error: command line arguments.\n");
      return 1;
    }

    max_number_adult = atoi(argv[1]);
    max_number_days = atoi(argv[2]);
    if( max_number_days <= 0)
        max_number_days = 1;
    max_number_households = max_number_adult/5;
    // max_number_workplaces=max_number_adult/100;

    int max_num_employee= 100;
    max_number_workplaces= max_number_adult / max_num_employee;


    unsigned long long  i;
    unsigned long long  j;
    unsigned long long h_numberOfInfected=0;
    int num_infected=10;

    printf( "start allocation \n" );

    adultAgents = (struct entity *)malloc(sizeof(struct entity)*max_number_adult);
    memset(adultAgents, 0,  (sizeof(struct entity)*max_number_adult) );


    infected_individuals = (unsigned long long  *) malloc(sizeof(unsigned long long )*max_number_adult);
    memset(infected_individuals, 0,  (sizeof(unsigned long long )*max_number_adult) );

    // set up our day update list
    dayUpdateList = (struct list_day_node*)malloc(sizeof(struct list_day_node)*max_number_days);

    printf( "start allocation on device \n" );

    cudaMalloc((void **) &d_adultAgents, sizeof(struct entity)*max_number_adult );

    cudaMalloc((void **) &d_houseHolds, sizeof(struct houseHold)*max_number_households );
    cudaMalloc((void **) &d_workPlaces, sizeof(struct workPlaces ) * max_number_workplaces);

    cudaMalloc((void **) &d_infected_individuals, sizeof(unsigned long long  ) * ( max_number_adult));

    // allocate the dayUpdateList on GPU
    cudaMalloc((void **) &d_dayUpdateList, sizeof(struct list_day_node)*max_number_days);

    printf( "finish allocation \n" );

    cudaMemset(d_adultAgents, 0, sizeof(struct entity)*max_number_adult);
    //  cudaMemset(d_childAgents, 0, sizeof(struct entity)*max_number_child);
    cudaMemset(d_houseHolds, 0, sizeof(struct houseHold)*max_number_households);
    cudaMemset(d_workPlaces, 0, sizeof(struct workPlaces)*max_number_workplaces);
    cudaMemset(d_infected_individuals, 0, sizeof(unsigned long long )* ( max_number_adult));

    cudaMemcpyToSymbol(dev_max_number_adult, &max_number_adult, sizeof(unsigned long long ));
    // cudaMemcpyToSymbol(dev_max_number_child, &max_number_child, sizeof(unsigned long long ));
    cudaMemcpyToSymbol(dev_max_number_house, &max_number_households, sizeof(unsigned long long ));
    float house_transmition = 0.47;
    float place_transmition = 0.94;
    float community_transmition = 0.075;
    cudaMemcpyToSymbol(house_trans, &house_transmition , sizeof(float));
    cudaMemcpyToSymbol(place_trans, &place_transmition, sizeof(float));
    cudaMemcpyToSymbol(comm_trans, &community_transmition, sizeof(float));

    for(j = 0;  j < num_infected; j++) {

        infected_individuals[j]=rand() % max_number_adult ;
        printf( "infected_individuals id is = %llu : \n", infected_individuals[j]);

    }

    int blocks_num;
    int startpoint=0;

    int houseType = 5;
    int residentStrttPoint=0;

    //blocks_num = (max_number_adult) / BLOCK_SIZE;
    blocks_num = (max_number_adult) / BLOCK_SIZE;

    dim3 grid(blocks_num, 1, 1);
    dim3 threads(BLOCK_SIZE, 1, 1);
    printf( "number of blocks is: %i \n", blocks_num );

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

    printf( "******** before calling the function ***** \n");

    kernel_generate_household << <grid, threads>> > ( startpoint,  houseType, residentStrttPoint, d_adultAgents ,d_houseHolds);


    printf( "******** after calling the function ***** \n");
    gpuErrchk(cudaDeviceSynchronize());


    blocks_num = max_number_workplaces / BLOCK_SIZE;
    printf( "number of blocks for work places is: %i \n", blocks_num );
    dim3 grid2(blocks_num, 1, 1);
    dim3 threads2(BLOCK_SIZE, 1, 1);
    kernel_generate_workplace << <grid2, threads2>> > ( max_num_employee, d_adultAgents, d_workPlaces);
    //move infected individuals

    gpuErrchk(cudaDeviceSynchronize());

    //update infected individuals status
    dim3 grid3(1, 1, 1);
	dim3 threads3(1, 1, 1);
    kernel_update_infected << <grid3, threads3>> > (infected_individuals[0],
        infected_individuals[1], infected_individuals[2], infected_individuals[3],
        infected_individuals[4], infected_individuals[5], infected_individuals[6],
        infected_individuals[7],infected_individuals[8], infected_individuals[9],
        d_adultAgents);
    gpuErrchk(cudaDeviceSynchronize());

    blocks_num = (max_number_adult) / BLOCK_SIZE;
    dim3 grid4(blocks_num, 1, 1);
    dim3 threads4(BLOCK_SIZE, 1, 1);

    //run the simulation
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int simulationDay;


    printf( "******** call the contact process ***** \n");
    //calclate force of infection
    //should be run for each day of simulation
    for (simulationDay=1; simulationDay <=2; simulationDay ++){
    kernel_calculate_contact_process << <grid4, threads4>> > ( d_infected_individuals,
        d_adultAgents, d_houseHolds, d_workPlaces, simulationDay, max_number_adult);
    }

    gpuErrchk(cudaDeviceSynchronize());


    /* check the total running time */
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    printf( "******** Total Running Time of Kernel = %0.5f ms \n", elapsedTime );

    //copy adults back to cpu
    //cudaMemcpy(adultAgents, d_adultAgents, sizeof(struct entity)*max_number_adult, cudaMemcpyDeviceToHost);
    cudaMemcpy(infected_individuals, d_infected_individuals, sizeof(unsigned long long )*max_number_adult, cudaMemcpyDeviceToHost);

    cudaMemcpyFromSymbol(&h_numberOfInfected, numberOfInfected, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&h_numberOfInfected, numberOfInfected, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("the number of infected copied back %llu \n", h_numberOfInfected);

    // copy the day list back to cpu
    cudaMemcpy(dayUpdateList, d_dayUpdateList, sizeof(struct list_day_node)*max_number_days, cudaMemcpyDeviceToHost);

    // set up our output FILE
    // file opening and error checking
    FILE *myfile = fopen(o_file_name, "w+"); // erases/creates the file
    if( !myfile ){
      printf("Error opening output file.\n");
    }
    else{
        fprintf(myfile, "Starting day by day output of simulation!\n\n");
        for(int i=0; i<simulationDay; i++){
            output_to_file(myfile,dayUpdateList[i]);
        }
    }
    fclose(myfile);

    /* Clean up memory */

    cudaFree(d_adultAgents);
    cudaFree(d_houseHolds);
    cudaFree(d_workPlaces);
    cudaFree(d_infected_individuals);


    free(adultAgents);
    free(infected_individuals);


    return 0;
}
