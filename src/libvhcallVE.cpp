#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <libvhcall.h>

#include "Geometry.hpp"

vhcall_handle vh = NULL;

// work
int numValidElements_;
int externalToLocalMap_size_;
int sendList_size_;
int receiveList_size_;
local_int_t totalToBeSent_;

extern "C" void
vhcallVE_SetupHalo(local_int_t *localNumberOfRows, int *numValidElements,
                   local_int_t *validRank, local_int_t *validi,
                   global_int_t *validcurIndex, int *externalToLocalMap_size,
                   int *sendList_size, int *receiveList_size,
                   local_int_t *totalToBeSent)
{
   int64_t sym;
   long result;

   size_t isize;
   void *ip;
   char *ipp;

   isize = sizeof(local_int_t) + sizeof(int) + sizeof(local_int_t)*(*numValidElements)*2 + sizeof(global_int_t)*(*numValidElements);
   ip = malloc(isize);

   ipp = (char *)ip;
   memcpy(ipp, localNumberOfRows, sizeof(local_int_t));                      ipp = ipp + sizeof(local_int_t);
   memcpy(ipp, numValidElements,  sizeof(int));                              ipp = ipp + sizeof(int);
   memcpy(ipp, validRank,         sizeof(local_int_t)*(*numValidElements));  ipp = ipp + sizeof(local_int_t)*(*numValidElements);
   memcpy(ipp, validi,            sizeof(local_int_t)*(*numValidElements));  ipp = ipp + sizeof(local_int_t)*(*numValidElements);
   memcpy(ipp, validcurIndex,     sizeof(global_int_t)*(*numValidElements)); ipp = ipp + sizeof(global_int_t)*(*numValidElements);

   size_t osize;
   void *op;
   char *opp;

   osize = sizeof(int)*3 + sizeof(local_int_t);
   op = malloc(osize);

   if (!vh) {
     vh = vhcall_install(getenv("LIBVHCALLVH"));
     if(vh == (vhcall_handle)-1){
       perror("Error occurred on vhcall_install. Does environment variable LIBVHCALLVH set correctly?");
       exit(EXIT_FAILURE);
     }
   }

   sym = vhcall_find(vh, "vhcallVH_SetupHalo");
   if(sym == (int64_t)-1){
      perror("Error occurred on vhcall_find. Does the function vhcallVH_SetupHalo exists?");
      exit(EXIT_FAILURE);
   }

   result = vhcall_invoke(sym, ip, isize, op, osize);
   if(result == -1){
      perror("Error occurred on vhcall_invoke for vhcallVH_SetupHalo.");
      exit(EXIT_FAILURE);
   }

   opp = (char *)op;
   memcpy(externalToLocalMap_size, opp, sizeof(int));         opp = opp + sizeof(int);
   memcpy(sendList_size,           opp, sizeof(int));         opp = opp + sizeof(int);
   memcpy(receiveList_size,        opp, sizeof(int));         opp = opp + sizeof(int);
   memcpy(totalToBeSent,           opp, sizeof(local_int_t)); opp = opp + sizeof(local_int_t);


   // work
   numValidElements_        = *numValidElements;
   externalToLocalMap_size_ = *externalToLocalMap_size;
   sendList_size_           = *sendList_size;
   receiveList_size_        = *receiveList_size;
   totalToBeSent_           = *totalToBeSent;

   free(ip);
   free(op);

   return;
}

extern "C" void
vhcallVE_Copyout(void *op, size_t osize, local_int_t **mtxIndL,
                 local_int_t **elementsToSend, int **neighbors,
                 local_int_t **receiveLength, local_int_t **sendLength)
{
   int64_t sym;
   long result;

   sym = vhcall_find(vh, "vhcallVH_CopyOut");
   if(sym == (int64_t)-1){
      perror("Error occurred on vhcall_find. Does the function vhcallVH_CopyOut exists?");
      exit(EXIT_FAILURE);
   }

   result = vhcall_invoke(sym, NULL, 0, op, osize);
   if(result == -1){
      perror("Error occurred on vhcall_invoke for vhcallVH_CopyOut.");
      exit(EXIT_FAILURE);
   }

   char *opp;

   opp = (char *)op;
   (*mtxIndL)        = (local_int_t *) opp; opp = opp + sizeof(local_int_t)*(numValidElements_);
   (*elementsToSend) = (local_int_t *) opp; opp = opp + sizeof(local_int_t)*(totalToBeSent_);
   (*neighbors)      = (int *) opp;         opp = opp + sizeof(int)*(sendList_size_);
   (*receiveLength)  = (local_int_t *) opp; opp = opp + sizeof(local_int_t)*(receiveList_size_);
   (*sendLength)     = (local_int_t *) opp; opp = opp + sizeof(local_int_t)*(sendList_size_);

   vh  = vhcall_uninstall(vh);

   return;
}

extern "C" void
vhcallVE_Hyperplane(local_int_t nrows, int maxNonzerosInRow,
                    local_int_t *mtxIndL_, local_int_t *nonzerosInRow,
                    int *icolor)
{

   int64_t sym;
   long result;

   size_t isize;
   void *ip;
   char *ipp;

   isize = sizeof(local_int_t) + sizeof(int) + sizeof(local_int_t)*(nrows*maxNonzerosInRow) + sizeof(local_int_t)*nrows;
   ip = malloc(isize);

   ipp = (char *)ip;
   memcpy(ipp, &nrows,            sizeof(local_int_t));                       ipp = ipp + sizeof(local_int_t);
   memcpy(ipp, &maxNonzerosInRow, sizeof(int));                               ipp = ipp + sizeof(int);
   memcpy(ipp, mtxIndL_,          sizeof(local_int_t)*nrows*maxNonzerosInRow);ipp = ipp + sizeof(local_int_t)*nrows*maxNonzerosInRow;
   memcpy(ipp, nonzerosInRow,     sizeof(local_int_t)*(nrows));               ipp = ipp + sizeof(local_int_t)*nrows;

   size_t osize;
   void *op;
   char *opp;

   osize = sizeof(int)*nrows + sizeof(int);
   op = malloc(osize);

   if (!vh) {
     vh = vhcall_install(getenv("LIBVHCALLVH"));
     if(vh == (vhcall_handle)-1){
       perror("Error occurred on vhcall_install. Does environment variable LIBVHCALLVH set correctly?");
       exit(EXIT_FAILURE);
     }
   }

   sym = vhcall_find(vh, "vhcallVH_Hyperplane");
   if(sym == (int64_t)-1){
      perror("Error occurred on vhcall_find. Does the function vhcallVH_Hyperplane exist?");
      exit(EXIT_FAILURE);
   }

   result = vhcall_invoke(sym, ip, isize, op, osize);
   if(result == -1){
      perror("Error occurred on vhcall_invoke for vhcallVH_MultiColor.");
      exit(EXIT_FAILURE);
   }

   opp = (char *)op;
   memcpy(icolor,           opp, sizeof(int)*nrows);   opp = opp + sizeof(int)*nrows;

   free(ip);
   free(op);

   return;
}

extern "C" void
vhcallVE_GetPerm(local_int_t n, int maxcolor, int *icolor, local_int_t *perm,
                 local_int_t *icptr)
{

   int64_t sym;
   long result;

   size_t isize = 0;
   void *ip = NULL;
   char *ipp;

   //isize = sizeof(local_int_t) + sizeof(int) + sizeof(int)*n;
   //ip = malloc(isize);
   //
   //ipp = ip;
   //memcpy(ipp, &n,                sizeof(local_int_t));                       ipp = ipp + sizeof(local_int_t);
   //memcpy(ipp, &maxcolor,         sizeof(local_int_t));                       ipp = ipp + sizeof(int);
   //memcpy(ipp, icolor,            sizeof(int)*n);                             ipp = ipp + sizeof(int)*n;
   ip = (void *)&maxcolor;
   isize = sizeof(int);
   
   size_t osize;
   void *op;
   char *opp;

   osize = sizeof(local_int_t)*n + sizeof(local_int_t)*(maxcolor+1);
   op = malloc(osize);

   if (!vh) {
     vh = vhcall_install(getenv("LIBVHCALLVH"));
     if(vh == (vhcall_handle)-1){
       perror("Error occurred on vhcall_install. Does environment variable LIBVHCALLVH set correctly?");
       exit(EXIT_FAILURE);
     }
   }

   sym = vhcall_find(vh, "vhcallVH_GetPerm");
   if(sym == (int64_t)-1){
      perror("Error occurred on vhcall_find. Does the function vhcallVH_GetPerm exist?");
      exit(EXIT_FAILURE);
   }

   result = vhcall_invoke(sym, ip, isize, op, osize);
   if(result == -1){
      perror("Error occurred on vhcall_invoke for vhcallVH_MultiColor.");
      exit(EXIT_FAILURE);
   }

   opp = (char *)op;
   memcpy(perm,       opp, sizeof(local_int_t)*n);              opp = opp + sizeof(local_int_t)*n;
   memcpy(icptr,      opp, sizeof(local_int_t)*(maxcolor+1));   opp = opp + sizeof(local_int_t)*(maxcolor+1);

   //free(ip);
   free(op);

   return;
}

