/* Instructions 
 * File: pth_mult_mutex.c
 * Compile: gcc -g -Wall -o pth_mult_mutex pth_mult_mutex.c -lpthread
 * Usage: ./pth_mult_mutex <thread_count>
 * Section 4.9.3
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "rand.h"
#include "timer.h"

const int MAX_KEY = 100000000;
int act_keys = 100;

const int IN_LIST = 1;
const int EMPTY_LIST = -1;
const int END_OF_LIST = 0;

struct list_node_s {
   int    data;
   pthread_mutex_t mutex;
   struct list_node_s* next;
};

struct list_node_s* head = NULL;  
pthread_mutex_t head_mutex;
int thread_count;
int total_ops;
double insert_p;
double search_p;
double delete_p;
pthread_mutex_t count_mutex;
int member_total=0, insert_total=0, delete_total=0;

/* Setup */
void Usage(char* prog_name);
void Get_input(int* inserts_in_main_p);

/* Thread function */
void* Thread_work(void* rank);

/* List operations */
void Init_ptrs(struct list_node_s** curr_pp, struct list_node_s** pred_pp);
int Advance_ptrs(struct list_node_s** curr_pp, 	struct list_node_s** pred_pp);
int Insert(int value);
void Print(void);
int Member(int value);
int Delete(int value);
void Free_list(void);
int Is_empty(void);

int main(int argc, char* argv[]) {
   long i; 
   int key, success, attempts;
   pthread_t* thread_handles;
   int inserts_in_main;
   unsigned seed = 1;
   double start, finish;
   if (argc != 2) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   Get_input(&inserts_in_main);
   i = attempts = 0;
   pthread_mutex_init(&head_mutex, NULL);
   while ( i < inserts_in_main && attempts < 2*inserts_in_main ) {
      key = my_rand(&seed) % MAX_KEY;
      success = Insert(key);
      attempts++;
      if (success) i++;
   }
   printf("Inserted %ld keys in empty list\n", i);
   thread_handles = malloc(thread_count*sizeof(pthread_t));
   pthread_mutex_init(&count_mutex, NULL);

   GET_TIME(start);
   for (i = 0; i < thread_count; i++)
      pthread_create(&thread_handles[i], NULL, Thread_work, (void*) i);

   for (i = 0; i < thread_count; i++)
      pthread_join(thread_handles[i], NULL);
   GET_TIME(finish);
   Print();	
   printf("Elapsed time = %e seconds\n", finish - start);
   printf("Total ops = %d\n", total_ops);
   printf("member ops = %d\n", member_total);
   printf("insert ops = %d\n", insert_total);
   printf("delete ops = %d\n", delete_total);
   Free_list();
   pthread_mutex_destroy(&head_mutex);
   pthread_mutex_destroy(&count_mutex);
   free(thread_handles);
   return 0;
}  /* main */

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count>\n", prog_name);
   exit(0);
}  /* Usage */

void Get_input(int* inserts_in_main_p) {
   printf("Keys:\n");
   scanf("%d", inserts_in_main_p);
   printf("Ops Number:\n");
   scanf("%d", &total_ops);
   printf("Percent of Ops in searches? (between 0 and 1)\n");
   scanf("%lf", &search_p);
   printf("Percent of Ops in inserts? (between 0 and 1)\n");
   scanf("%lf", &insert_p);
   delete_p = 1.0 - (search_p + insert_p);
}  /* Get_input */

void Init_ptrs(struct list_node_s** curr_pp, struct list_node_s** pred_pp) {
   *pred_pp = NULL;
   pthread_mutex_lock(&head_mutex);
   *curr_pp = head;
   if (*curr_pp != NULL)
      pthread_mutex_lock(&((*curr_pp)->mutex));
}  /* Init_ptrs */

int Advance_ptrs(struct list_node_s** curr_pp, struct list_node_s** pred_pp) {
   int rv = IN_LIST;
   struct list_node_s* curr_p = *curr_pp;
   struct list_node_s* pred_p = *pred_pp;
   if (curr_p == NULL) {
      if (pred_p == NULL) {
         /* At head of list */
         pthread_mutex_unlock(&head_mutex);
         return EMPTY_LIST;
       } else {  /* Not at head of list */
         return END_OF_LIST;
       }
   } else { // *curr_pp != NULL
      if (curr_p->next != NULL)
         pthread_mutex_lock(&(curr_p->next->mutex));
      else
         rv = END_OF_LIST;
      if (pred_p != NULL)
         pthread_mutex_unlock(&(pred_p->mutex));
      else
         pthread_mutex_unlock(&head_mutex);
      *pred_pp = curr_p;
      *curr_pp = curr_p->next;
      return rv;
   }
}  /* Advance_ptrs */

int Insert(int value) {
   struct list_node_s* curr;
   struct list_node_s* pred;
   struct list_node_s* temp;
   int rv = 1;

   Init_ptrs(&curr, &pred);
   
   while (curr != NULL && curr->data < value) {
      Advance_ptrs(&curr, &pred);
   }

   if (curr == NULL || curr->data > value) {
      temp = malloc(sizeof(struct list_node_s));
      pthread_mutex_init(&(temp->mutex), NULL);
      temp->data = value;
      temp->next = curr;
      if (curr != NULL) 
         pthread_mutex_unlock(&(curr->mutex));
      if (pred == NULL) {
         // Inserting in head of list
         head = temp;
         pthread_mutex_unlock(&head_mutex);
      } else {
         pred->next = temp;
         pthread_mutex_unlock(&(pred->mutex));
      }
   } else { /* value in list */
      if (curr != NULL) 
         pthread_mutex_unlock(&(curr->mutex));
      if (pred != NULL)
         pthread_mutex_unlock(&(pred->mutex));
      else
         pthread_mutex_unlock(&head_mutex);
      rv = 0;
   }

   return rv;
}  /* Insert */

void Print(void) {
   struct list_node_s* temp; 
   printf("list = ");
   temp = head;
   while (temp != (struct list_node_s*) NULL) {
      printf("%d ", temp->data);
      temp = temp->next;
   }
   printf("\n");
}  /* Print */

int  Member(int value) {
   struct list_node_s *temp, *old_temp;
   pthread_mutex_lock(&head_mutex);
   temp = head;
   if (temp != NULL) pthread_mutex_lock(&(temp->mutex));
   pthread_mutex_unlock(&head_mutex);
   while (temp != NULL && temp->data < value) {
      if (temp->next != NULL) 
         pthread_mutex_lock(&(temp->next->mutex));
      old_temp = temp;
      temp = temp->next;
      pthread_mutex_unlock(&(old_temp->mutex));
   }

   if (temp == NULL || temp->data > value) {
      if (temp != NULL) 
         pthread_mutex_unlock(&(temp->mutex));
      return 0;
   } else { /* temp != NULL && temp->data <= value */
      pthread_mutex_unlock(&(temp->mutex));
      return 1;
   }
}  /* Member */

int Delete(int value) {
   struct list_node_s* curr;
   struct list_node_s* pred;
   int rv = 1;
   Init_ptrs(&curr, &pred);
   while (curr != NULL && curr->data < value) {
      Advance_ptrs(&curr, &pred);
   }
   if (curr != NULL && curr->data == value) {
      if (pred == NULL) { /* first element in list */
         head = curr->next;
         pthread_mutex_unlock(&head_mutex);
         pthread_mutex_unlock(&(curr->mutex));
         pthread_mutex_destroy(&(curr->mutex));
         free(curr);
      } else { /* pred != NULL */
         pred->next = curr->next;
         pthread_mutex_unlock(&(pred->mutex));
         pthread_mutex_unlock(&(curr->mutex));
         pthread_mutex_destroy(&(curr->mutex));
         free(curr);
      }
   } else { /* Not in list */
      if (pred != NULL)
         pthread_mutex_unlock(&(pred->mutex));
      if (curr != NULL)
         pthread_mutex_unlock(&(curr->mutex));
      if (curr == head)
         pthread_mutex_unlock(&head_mutex);
      rv = 0;
   }

   return rv;
}  /* Delete */

void Free_list(void) {
   struct list_node_s* current;
   struct list_node_s* following;

   if (Is_empty()) return;
   current = head; 
   following = current->next;
   while (following != NULL) {
      free(current);
      current = following;
      following = current->next;
   }
   free(current);
}  /* Free_list */

int  Is_empty(void) {
   if (head == NULL)
      return 1;
   else
      return 0;
}  /* Is_empty */

void* Thread_work(void* rank) {
   long my_rank = (long) rank;
   int i, val;
   double which_op;
   unsigned seed = my_rank + 1;
   int my_member=0, my_insert=0, my_delete=0;
   int ops_per_thread = total_ops/thread_count;

   for (i = 0; i < ops_per_thread; i++) {
      which_op = my_drand(&seed);
      val = my_rand(&seed) % MAX_KEY;
      if (which_op < search_p) {
         Member(val);
         my_member++;
      } else if (which_op < search_p + insert_p) {
         Insert(val);
         my_insert++;
      } else { /* delete */
         Delete(val);
         my_delete++;
      }
   }  /* for */
   pthread_mutex_lock(&count_mutex);
   member_total += my_member;
   insert_total += my_insert;
   delete_total += my_delete;
   pthread_mutex_unlock(&count_mutex);
   return NULL;
}  /* Thread_work */