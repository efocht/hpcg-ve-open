#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include "type.h"

extern "C" void SetupHalo(local_int_t localNumberOfRows, int numValidElements, local_int_t *validRank,
                          local_int_t *validi, global_int_t *validcurIndex,
                          int *externalToLocalMap_size, int *sendList_size, int *receiveList_size, local_int_t *totalToBeSent,
                          local_int_t **mtxIndL, local_int_t **elementsToSend, int **neighbors, local_int_t **receiveLength, local_int_t **sendLength){

  // Scan global IDs of the nonzeros in the matrix.  Determine if the column ID matches a row ID.  If not:
  // 1) We call the ComputeRankOfMatrixRow function, which tells us the rank of the processor owning the row ID.
  //  We need to receive this value of the x vector during the halo exchange.
  // 2) We record our row ID since we know that the other processor will need this value from us, due to symmetry.

  std::map< int, std::vector< global_int_t> > sendList, receiveList;
  typedef std::map< int, std::vector< global_int_t> >::iterator map_iter;
  typedef std::vector<global_int_t>::iterator set_iter;
  std::map< local_int_t, local_int_t > externalToLocalMap;

  for (local_int_t i=0; i< numValidElements; i++) {
    int rank =  validRank[i];
    receiveList[rank].push_back(validcurIndex[i]);
    sendList[rank].push_back(validi[i]); // Matrix symmetry means we know the neighbor process wants my value
  }
  for (map_iter curNeighbor = receiveList.begin(); curNeighbor != receiveList.end(); ++curNeighbor) {
    sort((curNeighbor->second).begin(), (curNeighbor->second).end());
    (curNeighbor->second).erase(unique((curNeighbor->second).begin(), (curNeighbor->second).end()), (curNeighbor->second).end());
  }
  for (map_iter curNeighbor = sendList.begin(); curNeighbor != sendList.end(); ++curNeighbor) {
    sort((curNeighbor->second).begin(), (curNeighbor->second).end());
    (curNeighbor->second).erase(unique((curNeighbor->second).begin(), (curNeighbor->second).end()), (curNeighbor->second).end());
  }

  // Count number of matrix entries to send and receive
  (*totalToBeSent) = 0;
  for (map_iter curNeighbor = sendList.begin(); curNeighbor != sendList.end(); ++curNeighbor) {
    (*totalToBeSent) += (curNeighbor->second).size();
  }
  local_int_t totalToBeReceived = 0;
  for (map_iter curNeighbor = receiveList.begin(); curNeighbor != receiveList.end(); ++curNeighbor) {
    totalToBeReceived += (curNeighbor->second).size();
  }

  // Build the arrays and lists needed by the ExchangeHalo function.
  (*elementsToSend) = new local_int_t[(*totalToBeSent)];
  (*neighbors) = new int[sendList.size()];
  (*receiveLength) = new local_int_t[receiveList.size()];
  (*sendLength) = new local_int_t[sendList.size()];
  int neighborCount = 0;
  local_int_t receiveEntryCount = 0;
  local_int_t sendEntryCount = 0;
  for (map_iter curNeighbor = receiveList.begin(); curNeighbor != receiveList.end(); ++curNeighbor, ++neighborCount) {
    int neighborId = curNeighbor->first; // rank of current neighbor we are processing
    (*neighbors)[neighborCount] = neighborId; // store rank ID of current neighbor
    (*receiveLength)[neighborCount] = receiveList[neighborId].size();
    (*sendLength)[neighborCount] = sendList[neighborId].size(); // Get count if sends/receives
    for (set_iter i = receiveList[neighborId].begin(); i != receiveList[neighborId].end(); ++i, ++receiveEntryCount) {
      externalToLocalMap[*i] = localNumberOfRows + receiveEntryCount; // The remote columns are indexed at end of internals
    }
    for (set_iter i = sendList[neighborId].begin(); i != sendList[neighborId].end(); ++i, ++sendEntryCount) {
      (*elementsToSend)[sendEntryCount] = *i; // store local ids of entry to send
    }
  }

  (*mtxIndL) = new local_int_t[numValidElements];
  // Convert matrix indices to local IDs
  for (local_int_t i=0; i< numValidElements; i++) {
    (*mtxIndL)[i] = externalToLocalMap[validcurIndex[i]];
  }

  (*externalToLocalMap_size) = externalToLocalMap.size();
  (*sendList_size) = sendList.size();
  (*receiveList_size) = receiveList.size();

  return;
}

