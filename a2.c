#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

#include <pthread.h>
#include "mpi.h"

#define SHIFT_ROW 0
#define SHIFT_COL 1
#define DISP 1

#define MSG_SLAVE_EXIT 1
#define MSG_SLAVE_REPORT 2
#define MSG_SLAVE_ADJACENT 3

#define MAX_ITERATION 5
#define LOG_ITERATION 2

#define NUM_PORTS 5
#define AVAILABILITY_LIMIT 3
#define SHARED_ARRAY_SIZE 5

typedef struct port_status {
	time_t t;
	int availability;
} port_status;

// Array to represent shared array
typedef struct port_array {
	port_status* ports;
	int index; // Index to keep track of which index to update (circular queue - will go back to 0 when index == SHARED_ARRAY_SIZE)
} port_array;

// Logging data to be sent to base station
typedef struct log_data {
	time_t alert_time;
	int reporting_node_rank;
	int reporting_node_availability;
	int reporting_node_coord[2];
	int adj_nodes_rank[4];
	int adj_nodes_availability[4];
	int adj_nodes_coord[8];
	int no_adj_nodes;
	int dimensions[2];
	long tv_sec;
	long tv_nsec;
} log_data;

// Base data to be logged
typedef struct base_data {
	int i;
	time_t log_time;
	struct timespec recv_time;
	int* nearby_nodes;
	int no_nearby;
	int* avail_nodes;
	int no_avail;
} base_data;

int master_io(MPI_Comm world_comm, MPI_Comm comm);
int slave_io(MPI_Comm world_comm, MPI_Comm comm, int argc, char **argv);
void* thread_ports_func(void *pArg);
void* master_process_func(void *pArg);

int find_adj_nodes(int rank, int* dimension, int* adj_nodes);
void node_coord(int rank, int* dimension, int* coord);
int* helper_func(int no_adj_nodes, int* no_adj, int helper_array[][4], int* ret_int);
void formatTime(time_t timestamp, char *formattedTime, size_t maxLength);
void writeToFile(log_data data, base_data baseData, FILE* pFile);

unsigned int seed;
int* ports;
MPI_Datatype log_data_type;

int main(int argc, char **argv)
{

	// Initialize MPI
    int rank, size, provided, nrows, ncols;
    MPI_Comm new_comm;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Validate arguments provided
	if (argc == 3) {
		nrows = atoi (argv[1]);
		ncols = atoi (argv[2]);
		if( (nrows*ncols) != size - 1) {
			if( rank ==0) printf("ERROR: nrows*ncols)=%d * %d = %d != %d\n", nrows, ncols, nrows*ncols,size);
			MPI_Finalize();
			return 0;
		}
	} 


	// Create MPI Datatype for logging data to file
	int blocklengths[11] = {1, 1, 1, 2, 4, 4, 8, 1, 2, 1, 1};
	MPI_Aint offsets[11];
	MPI_Datatype types[11] = {MPI_LONG, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_LONG, MPI_LONG};

	offsets[0] = offsetof(log_data, alert_time);
	offsets[1] = offsetof(log_data, reporting_node_rank);
	offsets[2] = offsetof(log_data, reporting_node_availability);
	offsets[3] = offsetof(log_data, reporting_node_coord);
	offsets[4] = offsetof(log_data, adj_nodes_rank);
	offsets[5] = offsetof(log_data, adj_nodes_availability);
	offsets[6] = offsetof(log_data, adj_nodes_coord);
	offsets[7] = offsetof(log_data, no_adj_nodes);
	offsets[8] = offsetof(log_data, dimensions);
	offsets[9] = offsetof(log_data, tv_sec);
	offsets[10] = offsetof(log_data, tv_nsec);

	MPI_Type_create_struct(11, blocklengths, offsets, types, &log_data_type);
	MPI_Type_commit(&log_data_type);

	// Split communicator into base station and grid stations
    MPI_Comm_split( MPI_COMM_WORLD,rank == size - 1, 0, &new_comm);
    if (rank == size - 1) {
		master_io( MPI_COMM_WORLD, new_comm );
	} else {
		slave_io( MPI_COMM_WORLD, new_comm, argc, argv);
	}
	
	MPI_Type_free(&log_data_type);
    MPI_Finalize();
    return 0;
}

/* This is the Base Station */
int master_io(MPI_Comm world_comm, MPI_Comm comm)
{
	int size, nslaves;
	MPI_Comm_size(world_comm, &size );
	nslaves = size - 1;
	
	// Create thread to send and receive mpi messages
	pthread_t tid;
	pthread_create(&tid, 0, master_process_func, &nslaves); // Create the thread
	pthread_join(tid, NULL); // Wait for the thread to complete.

    return 0;
}

/* This is the grid station */
int slave_io(MPI_Comm world_comm, MPI_Comm comm, int argc, char **argv)
{
	int ndims=2, world_size, size, my_rank, reorder, my_cart_rank, ierr, nrows, ncols, nbr_i_lo, nbr_i_hi;
	int nbr_j_lo, nbr_j_hi;
	MPI_Comm comm2D;
	int dims[ndims],coord[ndims];
	int wrap_around[ndims];
	int i;
	double adj_time, base_time, adj_time_max = 0, base_time_max = 0;
	struct timespec adj_time_start, adj_time_end, base_time_start, base_time_end;
	char buf[512] = {0};

	/* start up initial MPI environment */
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &my_rank);
	MPI_Comm_size(world_comm, &world_size);

    seed = time(NULL) * my_rank;

    /* process command line arguments for virtual topology dimensions */
	if (argc == 3) {
		nrows = atoi (argv[1]);
		ncols = atoi (argv[2]);
		dims[0] = nrows; /* number of rows */
		dims[1] = ncols; /* number of columns */
		if( (nrows*ncols) != size) {
			if( my_rank ==0) printf("ERROR: nrows*ncols)=%d * %d = %d != %d\n", nrows, ncols, nrows*ncols,size);
			MPI_Finalize(); 
			return 0;
		}
	} else {
		nrows=ncols=(int)sqrt(size);
		dims[0]=dims[1]=0;
	}

    /*************************************************************/
	/* create cartesian topology for processes */
	/*************************************************************/
	MPI_Dims_create(size, ndims, dims);
	if(my_rank==0)
		printf("PW[%d], CommSz[%d]: PEdims = [%d x %d] \n",my_rank,size,dims[0],dims[1]);
	
	/* create cartesian mapping */
	wrap_around[0] = wrap_around[1] = 0; /* periodic shift is .false. */
	reorder = 1;
	ierr =0;
	ierr = MPI_Cart_create(comm, ndims, dims, wrap_around, reorder, &comm2D);
	if(ierr != 0) printf("ERROR[%d] creating CART\n",ierr);

    /* find my coordinates in the cartesian communicator group */
	MPI_Cart_coords(comm2D, my_rank, ndims, coord);
	/* use my cartesian coordinates to find my rank in cartesian group*/
	MPI_Cart_rank(comm2D, coord, &my_cart_rank);
	/* get my neighbors; axis is coordinate dimension of shift */
	/* axis=0 ==> shift along the rows: P[my_row-1]: P[me] : P[my_row+1] */
	/* axis=1 ==> shift along the columns P[my_col-1]: P[me] : P[my_col+1] */

    MPI_Cart_shift( comm2D, SHIFT_ROW, DISP, &nbr_i_lo, &nbr_i_hi );
	MPI_Cart_shift( comm2D, SHIFT_COL, DISP, &nbr_j_lo, &nbr_j_hi );
	int action_list[4] = {nbr_j_lo, nbr_j_hi, nbr_i_lo, nbr_i_hi};
	
	// Get number of adjacent nodes
	int no_adj = 0;
	for (int i = 0; i < 4; i++) {
		if (action_list[i] > -1)
			no_adj++;
	}

	// Initialises logging data with preprocess information such as rank, coordinates, topology dimension
	log_data data;
	data.reporting_node_rank = my_cart_rank;
	data.reporting_node_coord[0] = coord[0];
	data.reporting_node_coord[1] = coord[1];
	data.no_adj_nodes = no_adj;
	data.dimensions[0] = dims[0];
	data.dimensions[1] = dims[1];

	// Get adjacent nodes coordinates and ranks into logging data
	int index = 0;
	for (int i = 0; i < 4; i++) {
		if (action_list[i] > -1) {
			data.adj_nodes_rank[index] = action_list[i];
			MPI_Cart_coords(comm2D, action_list[i], ndims, coord);
			data.adj_nodes_coord[index * 2] = coord[0];
			data.adj_nodes_coord[index * 2 + 1] = coord[1];
			index++;
		}
	}

	// Initialize shared array to log availability
	port_array shared_array;
	shared_array.ports = calloc(SHARED_ARRAY_SIZE, sizeof(port_status));
	shared_array.index = 0;

	// Shared array for thread ports
	ports = calloc(NUM_PORTS, sizeof(int));

	time_t now;
	int exit_flag = 0, exit_msg;

	pthread_t tid[NUM_PORTS];
	int threadNum[NUM_PORTS];

	MPI_Status exit_status;
	MPI_Request exit_request;

	// Test for any exit message from base station
	MPI_Irecv(&exit_msg, 1, MPI_INT, world_size - 1, MSG_SLAVE_EXIT, world_comm, &exit_request);
	MPI_Test(&exit_request, &exit_flag, &exit_status);
	while (exit_flag == 0) { // Carry out availability check and reporting until exit message is received

		// Fork	thread ports to simulate availability
		for (i = 0; i < NUM_PORTS; i++)
		{
			threadNum[i] = i;
			pthread_create(&tid[i], 0, thread_ports_func, &threadNum[i]);
		}

		// Join threads
		for(i = 0; i < NUM_PORTS; i++)
		{
				pthread_join(tid[i], NULL);
		}

		// Update Shared Array of avability
		time(&now);
		shared_array.ports[shared_array.index].t = now;
		shared_array.ports[shared_array.index].availability = 0;

		for (int i = 0; i < NUM_PORTS; i++) {
			shared_array.ports[shared_array.index].availability += ports[i];
		}

		int recv_data_response[4] = {-1, -1, -1, -1};

		struct timespec t1;

		MPI_Status status;
		int adj_flag;

		// Check if availability is below limit
		if (shared_array.ports[shared_array.index].availability <= AVAILABILITY_LIMIT) {

			// Get time for node to send and receive from adjacent nodes
			clock_gettime(CLOCK_MONOTONIC, &adj_time_start);

			// Send availability to adjacent nodes
			for (i = 0; i < 4; i++) {
				MPI_Send(&shared_array.ports[shared_array.index].availability, 1, MPI_INT, action_list[i], MSG_SLAVE_ADJACENT, comm2D);
			}

			int full_adj = 0;

			// Receive availability from adjacent nodes
			for (int i = 0; i < 4; i++) {
				adj_flag = 0;
				MPI_Iprobe(action_list[i], MSG_SLAVE_ADJACENT, comm2D, &adj_flag, &status);

				if (adj_flag && action_list[i] > -1) {
					MPI_Recv(&recv_data_response[i], 1, MPI_INT, action_list[i], MSG_SLAVE_ADJACENT, comm2D, &status);
					full_adj++;
				}
			}

			// Calculate time taken for 2 way commounication with adjacent nodes
			clock_gettime(CLOCK_MONOTONIC, &adj_time_end);
			adj_time = (adj_time_end.tv_sec - adj_time_start.tv_sec) * 1e9;
			adj_time = (adj_time + (adj_time_end.tv_nsec - adj_time_start.tv_nsec)) * 1e-9;
			adj_time_max = adj_time_max > adj_time ? adj_time_max : adj_time;

			// Test for exit message from base station to ensure message not sent to base station after exit message received
			MPI_Test(&exit_request, &exit_flag, &exit_status);

			if (exit_flag == 0) {

				// If all adjacent nodes are full, notify base station
				if (full_adj == no_adj) {
					sprintf(buf, "Node %d is Full. No available ports on adjacent nodes; Notifying Base node...\n", my_cart_rank);
					fputs(buf, stdout);

					// Update logging data
					data.alert_time = shared_array.ports[shared_array.index].t;
					data.reporting_node_availability = shared_array.ports[shared_array.index].availability;

					int index = 0;

					// Update logging data with adjacent nodes availability
					for (int i = 0; i < 4; i++) {
						if (action_list[i] > -1) {
							data.adj_nodes_availability[index] = recv_data_response[i];
							index++;

						}
					}

					// Send current time to base station
					clock_gettime(CLOCK_MONOTONIC, &t1);
					data.tv_sec = t1.tv_sec;
					data.tv_nsec = t1.tv_nsec;

					clock_gettime(CLOCK_MONOTONIC, &base_time_start);

					// Send logging data to base station
					MPI_Send(&data, 1, log_data_type, world_size - 1, MSG_SLAVE_REPORT, world_comm);

					// Receive response from base station
					int message[8];
					MPI_Recv(message, 8, MPI_INT, world_size - 1, MSG_SLAVE_REPORT, world_comm, &status);
					if (message[0] == -2) {
						sprintf(buf, "Node %d: Received from base no available nodes nearby\n", my_cart_rank);
						fputs(buf, stdout);
					} else {
						for (int k = 0; k < 8; k++) {
							if (message[k] > -1) {
								sprintf(buf, "Node %d: Received from base Available Nodes are %d\n", my_cart_rank, message[k]);
								fputs(buf, stdout);
							}
						}
					}

					// Calculate time taken for 2 way commounication with base station
					clock_gettime(CLOCK_MONOTONIC, &base_time_end);
					base_time = (base_time_end.tv_sec - base_time_start.tv_sec) * 1e9;
					base_time = (base_time + (base_time_end.tv_nsec - base_time_start.tv_nsec)) * 1e-9;
					base_time_max = base_time_max > base_time ? base_time_max : base_time;

				} else { // If not all adjacent nodes are full print available adjacent nodes
					sprintf(buf, "Node %d is Full. Available Nodes are ", my_cart_rank);
					fputs(buf, stdout);
					for (i = 0; i < 4; i++) {
						if (recv_data_response[i] == -1 && action_list[i] > -1) {
							sprintf(buf, "%d ", action_list[i]);
							fputs(buf, stdout);
						}
					}
					sprintf(buf, "\n");
					fputs(buf, stdout);
				}
			}

			
		}
		
		// Update shared array index
		shared_array.index = (shared_array.index + 1) % SHARED_ARRAY_SIZE;
		
		sleep(1);
	}

	sprintf(buf, "Node %d: Max Communication with adj: %f base: %f\n", my_cart_rank, adj_time_max, base_time_max);
	fputs(buf, stdout);

	// Notify base station of Exit
    sprintf(buf, "Exit notification from %d\n", my_cart_rank);
	MPI_Send(buf, strlen(buf) + 1, MPI_CHAR, world_size - 1, MSG_SLAVE_EXIT, world_comm);

	return 0;
}

// Thread to send and receive mpi messages for base station
void* master_process_func(void *pArg) {

	int* p = (int*)pArg;
	int nslave = *p;

	char       buf[256];
	log_data data;
    FILE *pFile;

	MPI_Status status;
	MPI_Request request;

    sprintf(buf, "log_file.txt");
    pFile = fopen(buf, "w");

	time_t now;

	// Array to keep track of which slave nodes have reported
	int* report_array_flag = calloc(nslave, sizeof(int));
	int no_avail;

	struct timespec t1;
	int no_reports = 0;

	// For a set amount of iterations check for report from slave nodes
	// Each iteration will check for reports nslave times (once for each slave node)
    for (int i = 0; i < MAX_ITERATION; i++) {

		// Reset report array flag every LOG_ITERATION iterations
		if (i % LOG_ITERATION == 0) {
			for (int k = 0; k < nslave; k++) {
				report_array_flag[k] = 0;
			}
		}

		// Check for report from slave nodes		
		for (int j = 0; j < nslave; j++) {

			int flag = 0;

			// Check if report has been received from slave node
			MPI_Iprobe(MPI_ANY_SOURCE, MSG_SLAVE_REPORT, MPI_COMM_WORLD, &flag, &status);

			// If report has been received, log report and send available nodes
			if (flag) {

				// Keep track of number of reports for tabulation analysis
				no_reports++;

				// Get current time to log communication time
				clock_gettime(CLOCK_MONOTONIC, &t1);

				// Receive report from slave node
				MPI_Recv(&data, 1, log_data_type, MPI_ANY_SOURCE, MSG_SLAVE_REPORT, MPI_COMM_WORLD, &status);

				// Update base data with report information only provided by base station
				time(&now);
				base_data baseData;
				baseData.i = i;
				baseData.log_time = now;
				baseData.recv_time = t1;

				int no_adj[4];
				int helper_array[4][4];
				int temp;

				// Get nearby (adj's adj) nodes of reporting node
				for (int k = 0; k < data.no_adj_nodes; k++) {
				    no_adj[k] = find_adj_nodes(data.adj_nodes_rank[k], data.dimensions, helper_array[k]);
				}

				// Get non repeating set of nearby nodes
				int* nearby_nodes = helper_func(data.no_adj_nodes, no_adj, helper_array, &temp);

				int nearby[8];
				int counter = 0;
				for (int k = 0; k < temp; k++) {
				    if (k != data.reporting_node_rank && nearby_nodes[k] == 1) { // Check if node is not reporting node and is nearby
						nearby[counter] = k;
						counter++;
				    }
				}

				// Update base data with nearby nodes
				baseData.no_nearby = counter;
				baseData.nearby_nodes = nearby;

				no_avail = 0;
				int avail_node[4];

				// Check if report has been received from adjacent nodes of reporting node
				for (int k = 0; k < data.no_adj_nodes; k++) {
				    if (report_array_flag[data.adj_nodes_rank[k]] == 0) {
				        avail_node[no_avail] = data.adj_nodes_rank[k];
				        no_avail++;
				    }
				}

				baseData.no_avail = no_avail;

				int message[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
				counter = 0;

				for (int k = 0; k < no_avail; k++) {
					no_adj[k] = find_adj_nodes(avail_node[k], data.dimensions, helper_array[k]);
				}

				// Get non repeating set of nearby nodes that are available
				nearby_nodes = helper_func(no_avail, no_adj, helper_array, &temp);

				// Get available nodes into an array to be sent to grid station and logged
				for (int k = 0; k < temp; k++) {
					if (k != data.reporting_node_rank && nearby_nodes[k] == 1) {
						message[counter] = k;
						counter++;
					} 
				}
				free(nearby_nodes);

				// If no available nodes, send -2 to grid station
				if (counter == 0) {
				    message[counter] = -2;
				}

				baseData.avail_nodes = message;

				// Log data to file
				writeToFile(data, baseData, pFile);

				// Update report array flag
				report_array_flag[data.reporting_node_rank] = 1;

				// Send available nodes to grid station
				MPI_Send(message, 8, MPI_INT, status.MPI_SOURCE, MSG_SLAVE_REPORT, MPI_COMM_WORLD);
			}

			sleep(0.1);
		}


		sleep(1);
	}

	// Send exit message to slave nodes
	int exit_msg = 1;
	sprintf(buf, "Notifying slave nodes to exit\n");
	fputs(buf, stdout);
	for (int j = 0; j < nslave; j++) {
		MPI_Isend(&exit_msg, 1, MPI_INT, j, MSG_SLAVE_EXIT, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
	}

	// Wait for exit message from slave nodes
	while (nslave > 0) {
		MPI_Recv( buf, 256, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		switch (status.MPI_TAG) {
			case MSG_SLAVE_EXIT: 
				// sprintf(buf, "Received Exit Notification From Node: %d\n", status.MPI_SOURCE);
				// fputs(buf, stdout);
				nslave--; 
				break;
			case MSG_SLAVE_REPORT: // Ignore reports from slave nodes after exit message has been sent
				int message[8];
				message[0] = -2;
				MPI_Send(message, 8, MPI_INT, status.MPI_SOURCE, MSG_SLAVE_REPORT, MPI_COMM_WORLD);
			break;
		}
	}

	sprintf(buf, "Total number of reports: %d\n", no_reports);
	fputs(buf, stdout);
	
	return 0;
}

// Thread to simulate ports availability
void* thread_ports_func(void *pArg) {

	int rank = *((int*)pArg);

	int random_flag = rand_r(&seed) % 2;
	ports[rank] = random_flag;

	return 0;
}

// Find adjacent nodes of grid station
int find_adj_nodes(int rank, int* dimension, int* adj_nodes) {

	int index = 0;
	int displacement[2] = {-1, 1};
	int grid_nodes = dimension[0] * dimension[1];

	int adj_rank = 0;

	// Calculate adjacent nodes based on rank given
	for (int i = 0; i < 4; i++) {
		if (i == 0 || i == 3) {
			adj_rank = rank + displacement[i / 2] * dimension[1];
			if (adj_rank >= 0 && adj_rank < grid_nodes) { // Check if adjacent node is within grid
				adj_nodes[index] = adj_rank;
				index++;
			} 
		} else {
			adj_rank = rank + displacement[i / 2];

			// Check if adjacent node is within grid and on same row as grid station
			if (adj_rank / dimension[1] == rank / dimension[1] && adj_rank >= 0 && adj_rank < grid_nodes) { 
				adj_nodes[index] = adj_rank;
				index++;
			}
		}
	}

	// returns number of adjacent nodes
	return index;
}

// Get coordinates of grid station
void node_coord(int rank, int* dimension, int* coord) {
	coord[0] = rank / dimension[1];
	coord[1] = rank % dimension[1];
}

// Get non repeating set of nearby nodes
int* helper_func(int no_adj_nodes, int* no_adj, int helper_array[][4], int* ret_int) {

	// Returns a list of nodes that is the size of the max adjacent node
	// The list contains 1 if the index is a nearby not and 0 if not

	int temp = -1;

	// Get max adjacent node
	for (int k = 0; k < no_adj_nodes; k++) {
		if (temp < helper_array[k][no_adj[k] - 1]) {
			temp = helper_array[k][no_adj[k] - 1];
		}
	}

	temp += 1;

	int* nearby_nodes = calloc(temp, sizeof(int));

	// Set nearby nodes to 1
	for (int k = 0; k < no_adj_nodes; k++) {
		for (int l = 0; l < no_adj[k]; l++) {
			if (helper_array[k][l] > -1) {
				nearby_nodes[helper_array[k][l]] = 1;
			}
		}
	}

	*ret_int = temp;
	return nearby_nodes;
}

// Format time to be logged
void formatTime(time_t timestamp, char *formattedTime, size_t maxLength) {
    struct tm timeinfo;
    localtime_r(&timestamp, &timeinfo); // Convert to local time

    strftime(formattedTime, maxLength, "%a %Y-%m-%d %H:%M:%S", &timeinfo);
}

// Write logging data to file
void writeToFile(log_data data, base_data baseData, FILE* pFile) {

	// Format time to be logged
    char formattedLogTime[256];
    char formattedAlertTime[256];
    formatTime(baseData.log_time, formattedLogTime, sizeof(formattedLogTime));
    formatTime(data.alert_time, formattedAlertTime, sizeof(formattedAlertTime));

	// Write logging data to file
    fprintf(pFile, "------------------------------------------------------------------------------------------------------------\n");
    fprintf(pFile, "Iteration: %d\nLogged Time: \t\t\t%s\n", baseData.i, formattedLogTime);
    fprintf(pFile, "Alert reported time: \t%s\nNumber of adjacent node: %d\n", formattedAlertTime, data.no_adj_nodes);
    fprintf(pFile, "Availability to be considered full: %d\n\n", AVAILABILITY_LIMIT);
    fprintf(pFile, "Reporting Node \tCoord \t\tPort Value \tAvailable Port\n");
    fprintf(pFile, "%d\t\t\t\t(%d, %d) \t\t\t\t%d \t\t%d\n", data.reporting_node_rank, data.reporting_node_coord[0], data.reporting_node_coord[1], NUM_PORTS, data.reporting_node_availability);
    fprintf(pFile, "\nAdjacent Nodes \tCoord \t\tPort Value \tAvailable Port\n");
    for (int k = 0; k < data.no_adj_nodes; k++) {
        fprintf(pFile, "%d\t\t\t\t(%d, %d) \t\t\t\t%d \t\t%d\n", data.adj_nodes_rank[k], data.adj_nodes_coord[k * 2], data.adj_nodes_coord[k * 2 + 1], NUM_PORTS, data.adj_nodes_availability[k]);
    }

    int coord[2];

    fprintf(pFile, "\nNearby Nodes \tCoord\n");

    for (int k = 0; k < baseData.no_nearby; k++) {
        node_coord(baseData.nearby_nodes[k], data.dimensions, coord);
        fprintf(pFile, "%d\t\t\t\t(%d, %d)\n", baseData.nearby_nodes[k], coord[0], coord[1]);
    }

    fprintf(pFile, "\nAvailable station nearby (no report received in last %d iterations):", LOG_ITERATION);

	// Print available nodes
    if (baseData.no_avail < 1) {
        fprintf(pFile, " None");
    } else {
        for (int k = 0; k < 8; k++) {
			if (baseData.avail_nodes[k] > -1) {
				fprintf(pFile, " %d", baseData.avail_nodes[k]);
			}
        }
    }

	// Calculate communication time
    double time_taken = (baseData.recv_time.tv_sec - data.tv_sec) * 1e9;
    time_taken = (time_taken + (baseData.recv_time.tv_nsec - data.tv_nsec)) * 1e-9;

    fprintf(pFile, "\nCommunication Time: %f\n", time_taken);
    fprintf(pFile, "Total Messages send between reporting node and base station: 2\n\n");
}
