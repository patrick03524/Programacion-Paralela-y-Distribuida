/* Instructions 
 * File: linked_list_parallel.c
 * Section 4.9.
 */

#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "rand.h"
#include <pthread.h>

struct list_node_s {
  int data;
  struct list_node_s* next;
  pthread_mutex_t mutex;
};

struct list_node_s* head = NULL;
int thread_count;
int total_ops;
double insert_p;
double search_p;
double delete_p;
pthread_rwlock_t rwlock;
pthread_mutex_t count_mutex;
int Member_count = 0, insert_count = 0, delete_count = 0;

/* List operations */
int Insert(int value);
void Print(void);
int Member(int value);
int Delete(int value);
void Free_list(void);
int Is_empty(void);

int Insert(int value) {
   struct list_node_s* curr = head;
   struct list_node_s* pred = NULL;
   struct list_node_s* temp;
   int rv = 1;
   while (curr != NULL && curr->data < value) {
      pred = curr;
      curr = curr->next;
   }
   if (curr == NULL || curr->data > value) {
      temp = malloc(sizeof(struct list_node_s));
      temp->data = value;
      temp->next = curr;
      if (pred == NULL)
         head = temp;
      else
         pred->next = temp;
   } else { /* value in list */
      rv = 0;
   }

   return rv;
}  /* Insert */

void Print(void) {
   struct list_node_s* temp;
   printf("lista = ");
   temp = head;
   while (temp != (struct list_node_s*) NULL) {
      printf("%d ", temp->data);
      temp = temp->next;
   }
   printf("\n");
}  /* Print */

int  Member(int value) {
   struct list_node_s* temp;
   temp = head;
   while (temp != NULL && temp->data < value)
      temp = temp->next;
   if (temp == NULL || temp->data > value) {
      return 0;
   } else {
      return 1;
   }
}  /* Member */

int Delete(int value) {
   struct list_node_s* curr = head;
   struct list_node_s* pred = NULL;
   int rv = 1;
   /* Member value */
   while (curr != NULL && curr->data < value) {
      pred = curr;
      curr = curr->next;
   }
   if (curr != NULL && curr->data == value) {
      if (pred == NULL) { /* first element in list */
         head = curr->next;
         free(curr);
      } else { 
         pred->next = curr->next;
         free(curr);
      }
   } else { /* Not in list */
      rv = 0;
   }

   return rv;
}  /* Delete */

void Free_list(void) {
   struct list_node_s* p;
   struct list_node_s* temp;
   if (Is_empty()) return;
   p = head; 
   temp = p->next;
   while (temp != NULL) {
      free(p);
      p = temp;
      temp = p->next;
   }
   free(p);
}  /* Free_list */

int  Is_empty(void) {
   if (head == NULL)
      return 1;
   else
      return 0;
}  /* Is_empty */