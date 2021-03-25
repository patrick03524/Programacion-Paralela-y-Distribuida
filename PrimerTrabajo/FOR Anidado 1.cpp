#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;

#define MAX 1000
double **A, x[MAX], y[MAX];

void initialize();
void print_();

void initialize(){
	/*	Initialize A	*/
	A = new double*[MAX];
	for(int i = 0; i<MAX; ++i){
		A[i] = new double[MAX];
	}
	int numero_aleatorio;
	for(int i = 0; i<MAX; ++i){
		for(int j = 0; j<MAX; ++j){
			numero_aleatorio = 1 + rand() % (101 - 1);  ///numeros entre 1-100
			A[i][j] = numero_aleatorio;
		}
	}
	/*	Initialize x	*/
	for(int i = 0; i<MAX; ++i){
		numero_aleatorio = 1 + rand() % (101 - 1);  ///numeros entre 1-100
		x[i] = numero_aleatorio;
	}
	/*	Assign y = 0	*/
	for(int i = 0; i<MAX; ++i){
		y[i] = 0;
	}	
}
	
void print_(){
	cout<<"MATRIZ A"<<endl;
	for(int i = 0; i<MAX; ++i){
		for(int j = 0; j<MAX; ++j){
			cout<<A[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	cout<<"ARRAY X"<<endl;
	for(int i = 0; i<MAX; ++i){
		cout<<x[i]<<" ";
	}
	cout<<endl;
	cout<<"ARRAY Y"<<endl;
	for(int i = 0; i<MAX; ++i){
		cout<<y[i]<<endl;
	}
	cout<<endl;
}
	
int main(int argc, char *argv[]) {
	srand (time(NULL));
	/* Initialize A and x, assign y = 0 */
	initialize();
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	start = std::chrono::high_resolution_clock::now();
	/*	First pair of loops	*/
	for(int i = 0; i<MAX; ++i){
		for(int j = 0; j<MAX; ++j){
			y[i] += A[i][j] * x[j];
		}
	}		
	end = std::chrono::high_resolution_clock::now();
	int64_t duration =
		std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
		.count();
	std::cout << duration <<" nanoseconds"<<endl;	/*	Print	*/
	//print_();
	
	return 0;
}

