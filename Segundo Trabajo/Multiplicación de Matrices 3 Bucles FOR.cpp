#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;

#define matrizA 1000
#define matrizB 1000
#define matrizC 1000

double **A, **B, **C;

void initialize();
void print_();

void initialize(){
	/*	Initialize A	*/
	A = new double*[matrizA];
	for(int i = 0; i<matrizA; ++i){
		A[i] = new double[matrizB];
	}
	int numero_aleatorio;
	for(int i = 0; i<matrizA; ++i){
		for(int j = 0; j<matrizB; ++j){
			numero_aleatorio = 1 + rand() % (101 - 1);  ///numeros entre 1-100
			A[i][j] = numero_aleatorio;
		}
	}
	/*	Initialize B	*/
	B = new double*[matrizB];
	for(int i = 0; i<matrizB; ++i){
		B[i] = new double[matrizC];
	}
	for(int i = 0; i<matrizB; ++i){
		for(int j =0; j<matrizC; ++j){
			numero_aleatorio = 1 + rand() % (101 - 1);  ///numeros entre 1-100
			B[i][j] = numero_aleatorio;
		}
	}
	/*	Initialize C	*/
	C = new double*[matrizA];
	for(int i = 0; i<matrizA; ++i){
		C[i] = new double[matrizC];
	}
}
	
void print_(){
	cout<<"MATRIZ A"<<endl;
	for(int i = 0; i<matrizA; ++i){
		for(int j = 0; j<matrizB; ++j){
			cout<<A[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	cout<<"ARRAY B"<<endl;
	for(int i = 0; i<matrizB; ++i){
		for(int j = 0; j<matrizC; ++j){
			cout<<B[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	cout<<"ARRAY RES"<<endl;
	for(int i = 0; i<matrizA; ++i){
		for(int j = 0; j<matrizC; ++j){
			cout<<C[i][j]<<" ";
		}
		cout<<endl;
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
	
	for(int i = 0; i<matrizA; ++i){
		for(int j = 0; j<matrizC; ++j){
			C[i][j] = 0;
			for(int k =0; k<matrizB; ++k){
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	
	end = std::chrono::high_resolution_clock::now();
	int64_t duration =
		std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
		.count();
	std::cout << duration <<" nanoseconds"<<endl;
	std::cout << (double)duration/1000000000 <<" "<<endl;	/*	Print	*/
	//print_();
	
	return 0;
}

