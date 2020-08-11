#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <string.h>

#include <sys/stat.h>

#include "type.h"

int externalToLocalMap_size;
int sendList_size;
int receiveList_size;
local_int_t totalToBeSent;

local_int_t *mtxIndL;
local_int_t *elementsToSend;
int *neighbors;
local_int_t *receiveLength;
local_int_t *sendLength;

// work
int numValidElements_;
int externalToLocalMap_size_;
int sendList_size_;
int receiveList_size_;
local_int_t totalToBeSent_;
static local_int_t save_nrows, *save_icolor;


void SetupHalo(local_int_t localNumberOfRows, int numValidElements, local_int_t *validRank,
               local_int_t *validi, global_int_t *validcurIndex,
               int *externalToLocalMap_size, int *sendList_size, int *receiveList_size, local_int_t *totalToBeSent,
               local_int_t **mtxIndL, local_int_t **elementsToSend, int **neighbors, local_int_t **receiveLength,
               local_int_t **sendLength);

long vhcallVH_SetupHalo(void *hdl, void *ip, size_t isize, void *op, size_t osize){
   local_int_t *localNumberOfRows;
   int *numValidElements;
   local_int_t *validRank;
   local_int_t *validi;
   global_int_t *validcurIndex;

   char *ipp;
   int i;

   ipp = ip;
   localNumberOfRows = (local_int_t *)  ipp; ipp = ipp + sizeof(local_int_t);
   numValidElements  = (int *)          ipp; ipp = ipp + sizeof(int);
   validRank         = (local_int_t *)  ipp; ipp = ipp + sizeof(local_int_t)*(*numValidElements);
   validi            = (local_int_t *)  ipp; ipp = ipp + sizeof(local_int_t)*(*numValidElements);
   validcurIndex     = (global_int_t *) ipp; ipp = ipp + sizeof(global_int_t)*(*numValidElements);

   SetupHalo(*localNumberOfRows, *numValidElements, validRank, validi, validcurIndex, // Input
             &externalToLocalMap_size, &sendList_size, &receiveList_size, &totalToBeSent,
             &mtxIndL, &elementsToSend, &neighbors, &receiveLength, &sendLength);

   char *opp;

   opp = op;
   memcpy(opp, &externalToLocalMap_size, sizeof(int));         opp = opp + sizeof(int);
   memcpy(opp, &sendList_size,           sizeof(int));         opp = opp + sizeof(int);
   memcpy(opp, &receiveList_size,        sizeof(int));         opp = opp + sizeof(int);
   memcpy(opp, &totalToBeSent,           sizeof(local_int_t)); opp = opp + sizeof(local_int_t);

   // work
   numValidElements_        = *numValidElements;
   externalToLocalMap_size_ = externalToLocalMap_size;
   sendList_size_           = sendList_size;
   receiveList_size_        = receiveList_size;
   totalToBeSent_           = totalToBeSent;

   return EXIT_SUCCESS;
}

long vhcallVH_CopyOut(void *hdl, void *ip, size_t isize, void *op, size_t osize){
   char *opp;

   opp = op;
   memcpy(opp, mtxIndL,        sizeof(local_int_t)*(numValidElements_)); opp = opp + sizeof(local_int_t)*(numValidElements_);
   memcpy(opp, elementsToSend, sizeof(local_int_t)*(totalToBeSent_));    opp = opp + sizeof(local_int_t)*(totalToBeSent_);
   memcpy(opp, neighbors,      sizeof(int)*(sendList_size_));            opp = opp + sizeof(int)*(sendList_size_);
   memcpy(opp, receiveLength,  sizeof(local_int_t)*(receiveList_size_)); opp = opp + sizeof(local_int_t)*(receiveList_size_);
   memcpy(opp, sendLength,     sizeof(local_int_t)*(sendList_size_));    opp = opp + sizeof(local_int_t)*(sendList_size_);

   return EXIT_SUCCESS;
}

#if 0
void MultiColor(local_int_t nrow, int maxNonzerosInRow, local_int_t *mtxIndL_, local_int_t *nonzerosInRow,
		int *icolor, int *totalColors);

long vhcallVH_MultiColor(void *hdl, void *ip, size_t isize, void *op, size_t osize)
{
  char *opp = (char *)op;
  char *ipp = (char *)ip;
  local_int_t nrows;
  int maxNonzerosInRow;
  local_int_t *mtxIndL_;
  local_int_t *nonzerosInRow;
  int *icolor, *totalColors;

  nrows            = *((local_int_t *)ipp); ipp = ipp + sizeof(local_int_t);
  maxNonzerosInRow = *((int *)ipp);         ipp = ipp + sizeof(int);
  mtxIndL_         = (local_int_t *)ipp;    ipp = ipp + sizeof(local_int_t)*nrows*maxNonzerosInRow;
  nonzerosInRow    = (local_int_t *)ipp;    ipp = ipp + sizeof(local_int_t)*nrows;

  icolor = (int *)opp;
  opp = opp + sizeof(int) * nrows;
  totalColors = (int *)opp;

  //printf("VHcall: before &totalColors= 0x%llx  (op=0x%llx, osize=%ld) \n", totalColors, op, osize);

  MultiColor(nrows, maxNonzerosInRow, mtxIndL_, nonzerosInRow, icolor, totalColors);

  //printf("VHcall: totalColors (0x%llx)\n", totalColors);
  save_nrows = nrows;
  save_icolor = malloc(sizeof(local_int_t) * nrows);
  memcpy((void *)save_icolor, (const void *)icolor, sizeof(local_int_t) * nrows);

  return EXIT_SUCCESS;
}
#endif

void Hyperplane(local_int_t nrow, int maxNonzerosInRow, local_int_t *mtxIndL_, local_int_t *nonzerosInRow,
		int *icolor);

long vhcallVH_Hyperplane(void *hdl, void *ip, size_t isize, void *op, size_t osize)
{
  char *opp = (char *)op;
  char *ipp = (char *)ip;
  local_int_t nrows;
  int maxNonzerosInRow;
  local_int_t *mtxIndL_;
  local_int_t *nonzerosInRow;
  int *icolor, *totalColors;

  nrows            = *((local_int_t *)ipp); ipp = ipp + sizeof(local_int_t);
  maxNonzerosInRow = *((int *)ipp);         ipp = ipp + sizeof(int);
  mtxIndL_         = (local_int_t *)ipp;    ipp = ipp + sizeof(local_int_t)*nrows*maxNonzerosInRow;
  nonzerosInRow    = (local_int_t *)ipp;    ipp = ipp + sizeof(local_int_t)*nrows;

  icolor = (int *)opp;
  opp = opp + sizeof(int) * nrows;

  Hyperplane(nrows, maxNonzerosInRow, mtxIndL_, nonzerosInRow, icolor);

  save_nrows = nrows;
  save_icolor = malloc(sizeof(local_int_t) * nrows);
  memcpy((void *)save_icolor, (const void *)icolor, sizeof(local_int_t) * nrows);

  return EXIT_SUCCESS;
}

#if 0
void  Hypercube(local_int_t nrow, int maxNonzerosInRow,
                local_int_t *mtxIndL_, local_int_t *nonzerosInRow,
                uint32_t *hcperm, int *icolor);

long vhcallVH_Hypercube(void *hdl, void *ip, size_t isize, void *op, size_t osize)
{
  char *opp = (char *)op;
  char *ipp = (char *)ip;
  local_int_t nrows, nx, ny, nz;
  int maxNonzerosInRow;
  local_int_t *mtxIndL_;
  local_int_t *nonzerosInRow;
  uint32_t *hcperm;
  int *icolor;

  nrows            = *((local_int_t *)ipp); ipp = ipp + sizeof(local_int_t);
  maxNonzerosInRow = *((int *)ipp);         ipp = ipp + sizeof(int);
  mtxIndL_         = (local_int_t *)ipp;    ipp = ipp + sizeof(local_int_t)*nrows*maxNonzerosInRow;
  nonzerosInRow    = (local_int_t *)ipp;    ipp = ipp + sizeof(local_int_t)*nrows;
  hcperm           = (local_int_t *)ipp;    ipp = ipp + sizeof(local_int_t)*nrows;

  icolor = (int *)opp;
  opp = opp + sizeof(int) * nrows;

  Hypercube(nrows, maxNonzerosInRow, mtxIndL_, nonzerosInRow, hcperm, icolor);

  save_nrows = nrows;
  save_icolor = malloc(sizeof(local_int_t) * nrows);
  memcpy((void *)save_icolor, (const void *)icolor, sizeof(local_int_t) * nrows);

  return EXIT_SUCCESS;
}
#endif

void GetPerm(local_int_t n, int maxcolor, int *icolor, local_int_t *perm, local_int_t *icptr);

long vhcallVH_GetPerm(void *hdl, void *ip, size_t isize, void *op, size_t osize)
{
  char *opp = (char *)op;
  char *ipp = (char *)ip;
  local_int_t n;
  int maxcolor;
  local_int_t *perm;
  local_int_t *icptr;
  int *icolor;

  //n         = *((local_int_t *)ipp); ipp = ipp + sizeof(local_int_t);
  maxcolor  = *((int *)ipp);         ipp = ipp + sizeof(int);
  //icolor    = (int *)ipp;            ipp = ipp + sizeof(int)*n;
  n = save_nrows;
  icolor = save_icolor;

  perm = (int *)opp;  opp = opp + sizeof(local_int_t) * n;
  icptr = (int *)opp; opp = opp + sizeof(local_int_t) * (maxcolor+1);

  GetPerm(n, maxcolor, icolor, perm, icptr);

  free(save_icolor);

  return EXIT_SUCCESS;
}

