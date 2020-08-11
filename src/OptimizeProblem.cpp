
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// Erich Focht : Version optimized for SX-Aurora
//
// ***************************************************
//@HEADER

/*!
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
#include <stdio.h>
#include <cstdlib>

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)
//EF//#include <iostream>
//EF//#include <mpi.h>
#include <fstream>
using namespace std;
using std::endl;
#include "hpcg.hpp"
#endif

#include <iostream>
#include <iomanip>
using namespace std;

#ifdef HPCG_DEBUG
#include <fstream>
static int first_call = 1;
#endif

static int mpi_rank;

extern "C"{
  void dgthr_(const local_int_t* n, const double* x, double* px, const local_int_t* p);
  void ssctr_(local_int_t* n, local_int_t* x, local_int_t* indx, local_int_t* y);

  void vhcallVE_Hyperplane(local_int_t nrows, int maxNonzerosInRow, local_int_t *mtxIndL_, local_int_t *nonzerosInRow,
                           int *icolor);
  void vhcallVE_GetPerm(local_int_t n, int maxcolor, int *icolor, local_int_t *perm, local_int_t *icptr);

#ifdef FTRACE
  int ftrace_region_begin(const char *id);
  int ftrace_region_end(const char *id);
#endif
}


static void Optimize_Hyperplane(SparseMatrix & A)
{
  OPT*        opt = (OPT *)A.optimizationData;
  ELL*        ell = opt->ell;
  local_int_t n   = A.localNumberOfRows;
  local_int_t maxNonzerosInRow = ell->m + 1;
  local_int_t *icolor = opt->icolor;

  ///////////////////////////////////////////////////////
  // Hyperplane method
  ///////////////////////////////////////////////////////
  
#ifdef VHCALL
  vhcallVE_Hyperplane(n, maxNonzerosInRow, A.mtxIndL[0], A.nonzerosInRow, icolor);
#else
  for(local_int_t i=0; i<n; i++) icolor[i] = 0; // initialization

  for(local_int_t i=0; i<n; i++){
    local_int_t m=0;
#pragma _NEC novector
    for(local_int_t jj=0; jj<A.nonzerosInRow[i]; jj++){
      local_int_t j = *(A.mtxIndL[i]+jj);
      if (j<i && m<icolor[j]) m=icolor[j];
    }
    icolor[i]=m+1;
  }
  // Hyperplanes lead to colors starting with 1.
  // Make color numbers start with 0.
  for(local_int_t i=0; i<n;i++)
    icolor[i]--;
#endif /* VHCALL */
}

//-------------------------------------------------
//-------------------------------------------------
//-------------------------------------------------
static void Optimize_Store_ELL_L_U_halo(SparseMatrix & A)
{
  OPT*        opt = (OPT *)A.optimizationData;
  ELL*        ell = opt->ell;
  local_int_t   n = A.localNumberOfRows;
  local_int_t maxNonzerosInRow = ell->m + 1;
  local_int_t  *icolor = opt->icolor;
  local_int_t        m = ell->m;
  local_int_t maxcolor = opt->maxcolor;
  local_int_t   *icptr = opt->icptr;

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
  //  Store the sparse matrix A in ELL format and the HALO matrix //
  //    (The sparse matrix A is permuted.)                        //
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

  local_int_t lda = ell->lda = n;
  HALO*      halo = opt->halo;

  local_int_t      nh = halo->nh   = A.numberOfExternalValues;
  local_int_t   *rows = halo->rows = new local_int_t[nh + maxcolor+1 + 2*maxcolor];
  local_int_t  *hcptr = halo->hcptr = rows + nh;
  local_int_t  *color_mL = opt->color_mL = hcptr + maxcolor + 1;
  local_int_t  *color_mU = opt->color_mU = color_mL + maxcolor;

  //local_int_t *itmp = new local_int_t[3*n+2*nh]; // array for temporary data
  local_int_t itmp[3*n+2*nh];
  local_int_t* colIndL = itmp;  // colindex full or lower matrix
  local_int_t* colIndU = colIndL + n; // coll index upper matrix
  local_int_t* hrows = colIndU + n; // flag marking halo rows
  local_int_t *hcolor = hrows + n;  // length: nh
  local_int_t* colIndH = hcolor + nh; // coll index halo matrix, length nh

#ifdef FTRACE
  ftrace_region_begin("LUH_1");
#endif
  //
  // Cound L U column widths of reordered and normal matrix
  //

  for(local_int_t i=0; i<n; i++) {
    colIndL[i]=0;
    colIndU[i]=0;
    hrows[i]=0;
  }

  if (m < 254 && n < (1<<24)) {                          // for small m we know we can fold the value 
    for(local_int_t jj=0; jj<maxNonzerosInRow; jj++){
#pragma _NEC ivdep
      for(local_int_t i=0; i<n; i++){
        local_int_t ijj = i*maxNonzerosInRow+jj;
        local_int_t j    = A.mtxIndL[0][ijj];
        if (j < 0) continue;
        local_int_t inew = opt->perm0[i];
        if (j >= n) {      // if j>=n then j points to neighbor region.
          A.mtxIndL[0][ijj] |= (hrows[inew]<<24);
          hrows[inew]++;
        } else if ( opt->perm0[j] > inew ){  // jnew > inew
          A.mtxIndL[0][ijj] |= (colIndU[i]<<24);
          colIndU[i]++;
        } else if ( opt->perm0[j] < inew ) {  // jnew < inew
          A.mtxIndL[0][ijj] |= (colIndL[i]<<24);
          colIndL[i]++;
        }
      }
    }
  } else {
    for(local_int_t jj=0; jj<maxNonzerosInRow; jj++){
#pragma _NEC ivdep
      for(local_int_t i=0; i<n; i++){
        local_int_t ijj = i*maxNonzerosInRow+jj;
        local_int_t j    = A.mtxIndL[0][ijj];
        if (j < 0) continue;
        local_int_t inew = opt->perm0[i];
        if (j >= n) {      // if j>=n then j points to neighbor region.
          hrows[inew]++;
        } else if ( opt->perm0[j] > inew ){  // jnew > inew
          colIndU[i]++;
        } else if ( opt->perm0[j] < inew ) {  // jnew < inew
          colIndL[i]++;
        }
      }
    }
  }

  //assert(A.localNumberOfNonzeros == anonzeros + hnonzeros);
#ifdef FTRACE
  ftrace_region_end("LUH_1");
  ftrace_region_begin("LUH_2");
#endif
  // count rows in halo matrix
  local_int_t nhrows = 0, max_indH = 0;
  for (local_int_t inew=0; inew<n; inew++) {
    if (hrows[inew] > 0) {
      rows[nhrows] = inew;
      nhrows++;
      if (hrows[inew] > max_indH)
        max_indH = hrows[inew];
    }
  }
  // this is the number of halo matrix rows!
  // confusing this with nh leads to wrong hcptr[]
  local_int_t nah = halo->nah = nhrows;
  
  for(local_int_t ih=0; ih<nah; ih++){
    local_int_t inew = rows[ih];
    hrows[inew] = ih;
    hcolor[ih] = icolor[opt->iperm0[inew]];
  }

  // count up halo colors to make a pointer array
  for (local_int_t ic=0; ic<=maxcolor; ic++)
    hcptr[ic] = 0;

  // width of upper part of matrix for each color
  int max_indL = 0, max_indU = 0;
  for (local_int_t ic=0; ic<maxcolor; ic++) {
    local_int_t ics       = icptr[ic];
    local_int_t ice       = icptr[ic+1];
    int maxL = 0, maxU = 0;
    for (local_int_t i=ics; i<ice; i++) {
      int wL = colIndL[opt->iperm0[i]];
      int wU = colIndU[opt->iperm0[i]];
      if (wL > maxL) maxL = wL;
      if (wU > maxU) maxU = wU;
    }
    color_mL[ic] = maxL;
    color_mU[ic] = maxU;
    if (color_mL[ic] > max_indL)
      max_indL = color_mL[ic];
    if (color_mU[ic] > max_indU)
      max_indU = color_mU[ic];
  }
  local_int_t   mL = ell->mL = max_indL;
  local_int_t   mU = ell->mU = max_indU;
  local_int_t   mh = halo->mh = max_indH;
  local_int_t ldah = halo->ldah = nah;
  m = ell->m = mL + mU;

  //if (mpi_rank == 0) {
  //  cout<<" color_mU (n="<<n<<")"<<endl;
  //  for (local_int_t ic=0; ic<maxcolor; ic++)
  //    cout<<fixed<<setw(5)<<ic<<"  color_mL="<<setw(4)<<opt->color_mL[ic]
  //        <<"  color_mU="<<opt->color_mU[ic]<<"   nc="<<icptr[ic+1]-icptr[ic]
  //        << endl;
  //}

#ifdef FTRACE
  ftrace_region_end("LUH_2");
  ftrace_region_begin("LUH_3");
#endif

  if (nah > 5000) {
    int hhtmp[256*(maxcolor)];
    for (int i=0; i<256*maxcolor; i++)
      hhtmp[i] = 0;
#pragma _NEC ivdep
    for (local_int_t ih=0; ih<nah; ih++) {
      int bin = ih % 256;
      int col = hcolor[ih];
      hhtmp[bin*maxcolor+col]++;
    }
    for (int i=0; i<256; i++)
      for (int ic=0; ic<maxcolor;ic++)
#pragma _NEC ivdep
        hcptr[ic+1] += hhtmp[i*maxcolor+ic];
  } else {
    for (local_int_t ih=0; ih<nah; ih++)
      hcptr[hcolor[ih]+1]++;
  }

  for (local_int_t ic=1; ic<=maxcolor; ic++)
    hcptr[ic] += hcptr[ic-1];

#ifdef FTRACE
  ftrace_region_end("LUH_3");
  ftrace_region_begin("LUH_4");
#endif
  
  if (m != mL + mU) {
    cout<<"!!! m != mL + mU"<<endl<<flush;
    m = ell->m = mL+mU;
  }

  // we don't need icolor any more
  delete [] icolor;
  
  ell->a     = new double[lda*(mL+mU) + nah*mh + nah];    // allocate multiple arrays in one step
  double *ah = halo->ah = ell->a + lda*(mL+mU);
  halo->v = ah + nah*mh;

  ell->ja    = new local_int_t[lda*(mL+mU) + nah*mh];   // allocate multiple arrays in one step
  local_int_t *jah = halo->jah = ell->ja + lda*(mL+mU);
  
  // initialization
  for(local_int_t j=0; j<mL+mU; j++){
    for(local_int_t i=0; i<n; i++){
      ell->a[i+lda*j]  = 0;
      ell->ja[i+lda*j] = i;            // 0 origin
    }
  }

#pragma omp parallel for
  for(local_int_t jh=0; jh<mh; jh++){
    for(local_int_t ih=0; ih<nah; ih++){
      halo->ah[ih+ldah*jh]  = 0;
      halo->jah[ih+ldah*jh] = ih;            // 0 origin
    }
  }
#ifdef FTRACE
  ftrace_region_end("LUH_4");
  ftrace_region_begin("LUH_5");
#endif
  for(local_int_t i=0; i<n; i++) {
    colIndL[i]=0;
    colIndU[i]=0;
  }
  for(local_int_t ih=0; ih<nah; ih++)
    colIndH[ih] = 0;

  if (maxNonzerosInRow >= 254 || n > (1<<24)) {
    for(local_int_t jj=0; jj<maxNonzerosInRow; jj++){
#pragma _NEC ivdep
      for(local_int_t i=0; i<n; i++){
        local_int_t ijj = i*maxNonzerosInRow+jj;
        if ( jj<A.nonzerosInRow[i] ){
          //local_int_t j    = *(A.mtxIndL[i]+jj);
          //double      aval = *(A.matrixValues[i]+jj);
          local_int_t j    = A.mtxIndL[0][ijj];
          double      aval = A.matrixValues[0][ijj];
          local_int_t inew = opt->perm0[i];
          local_int_t jnew;
          if (j<n) {              // only local elements
            jnew = opt->perm0[j];
            if ( jnew>inew ){
              // upper non-diagonal part of the sparse matrix A
              local_int_t kU = colIndU[i] + mL;
              colIndU[i]++;
              ell->ja[inew+lda*kU] = jnew;       // 0 origin
              ell->a[inew+lda*kU]  = aval;
            } else if( jnew<inew ) {
              // lower non-diagonal part of the sparse matrix A
              local_int_t kL = colIndL[i];
              colIndL[i]++;
              ell->ja[inew+lda*kL] = jnew;       // 0 origin
              ell->a[inew+lda*kL]  = aval;
            } else {  // diagonal part of the sparse matrix A
              opt->diag[inew] = aval;
              opt->idiag[inew] = 1/aval;
            }
          } else {     // if j>=n then j points neighbor region.
            jnew = j;
            local_int_t ih = hrows[inew];
            local_int_t kh = colIndH[ih];
            colIndH[ih]++;
            halo->jah[ih+ldah*kh] = j - n;       // 0 origin
            halo->ah[ih+ldah*kh]  = aval;
          }
        }
      }
    }
  } else {   // m < 254
#pragma _NEC ivdep
    for(local_int_t ijj=0; ijj<n*maxNonzerosInRow; ijj++){
      local_int_t i = ijj / maxNonzerosInRow;
      local_int_t jj = ijj % maxNonzerosInRow;
      local_int_t j    = A.mtxIndL[0][ijj];
      if (j >= 0) {
        local_int_t k = (j >> 24);
        j = j & 0xffffff;
        A.mtxIndL[0][ijj] = j;
        double      aval = A.matrixValues[0][ijj];
        local_int_t inew = opt->perm0[i];
        local_int_t jnew;
        if (j<n) {              // only local elements
          jnew = opt->perm0[j];
          if ( jnew>inew ){
            // upper non-diagonal part of the sparse matrix A
            local_int_t kU = k + mL;
            ell->ja[inew+lda*kU] = jnew;       // 0 origin
            ell->a[inew+lda*kU]  = aval;
          } else if( jnew<inew ) {
            // lower non-diagonal part of the sparse matrix A
            local_int_t kL = k;
            ell->ja[inew+lda*kL] = jnew;       // 0 origin
            ell->a[inew+lda*kL]  = aval;
          } else {  // diagonal part of the sparse matrix A
            opt->diag[inew] = aval;
            opt->idiag[inew] = 1/aval;
          }
        } else {     // if j>=n then j points neighbor region.
          jnew = j;
          local_int_t ih = hrows[inew];
          local_int_t kh = k;
          halo->jah[ih+ldah*kh] = j - n;       // 0 origin
          halo->ah[ih+ldah*kh]  = aval;
        }
      }
    }
  }

#ifdef FTRACE
  ftrace_region_end("LUH_5");
#endif

#if HPCG_DEBUG
  if (mpi_rank==0 && n<520) {
    cout<<"Matrix A (n="<<n<<", mU="<<mU<<", mL="<<mL<<")"<<endl;
    for(int i=0; i<n; i++) {
      cout<<"  i="<<fixed<<setw(3)<<i<<"   ";
      for(int j=0; j<mL+mU; j++) {
        int jj=ell->ja[i+lda*j];
        cout<<fixed<<setw(8);
        if (jj>n) cout<<"["<<jj<<"]";
        else cout<<jj<<"  ";
      }
      cout<<endl<<"           ";
      for(int j=0; j<m; j++)
        cout<<setprecision(1)<<scientific<<ell->a[i+lda*j]<<"  ";
      cout<<endl<<endl;
    }
  }

  if (mpi_rank==0 && n<520) {
    cout<<"Matrix AH (n="<<n<<", nh="<<nh<<")"<<endl;
    for(int ih=0; ih<nah; ih++) {
      cout<<"  ih="<<fixed<<setw(3)<<ih<<"   ";
      for(int jh=0; jh<mh; jh++)
        cout<<fixed<<setw(7)<<halo->jah[ih+ldah*jh]<<"  ";
      cout<<endl<<"r="<<fixed<<setw(3)<<rows[ih]<<" c="<<hcolor[ih]<<" ";
      for(int jh=0; jh<mh; jh++)
        cout<<setprecision(1)<<scientific<<halo->ah[ih+ldah*jh]<<"  ";
      cout<<endl<<endl;
    }
  }
  if (mpi_rank==0 && n<520) {
    cout<<"hcptr: (n="<<n<<")"<<endl;
    for(int ic=0; ic<maxcolor+1; ic++) {
      cout<<"  ic="<<fixed<<setw(3)<<ic<<"   hcptr="<<fixed<<setw(5)<<hcptr[ic]<<endl;
    }
  }
  if (mpi_rank==0 && n<520) {
    cout<<"elementsToSend (n="<<n<<", num="<<A.totalToBeSent<<")"<<endl;
    for(int ih=0; ih<A.totalToBeSent; ih++) {
      cout<<"  ih="<<fixed<<setw(3)<<ih<<"   row="<<fixed<<setw(5)<<A.elementsToSend[ih]<<endl;
    }
  }
#endif
}

int _CheckDone(SparseMatrix &A)
{
  OPT*  opt = (OPT*)(A.optimizationData);
  ELL*        ell = opt->ell;

  if (ell->a == nullptr) {
    Optimize_Store_ELL_L_U_halo(A);
  }
  return 0;
}

void Optimize_CheckDone(SparseMatrix &A)
{
  vcycle(_CheckDone, A);
}

void Optimize_ReplaceMatrixDiagonal(SparseMatrix &A, double *dv)
{
  OPT*  opt = (OPT*)(A.optimizationData);
  ELL*        ell = opt->ell;

  if (ell->a == nullptr) {
    cout<<"checkdone in ReplaceMatrixDiagonal for n=" << ell->n << endl;
    Optimize_CheckDone(A);
  }

  for (local_int_t i=0; i<A.localNumberOfRows; ++i) {
    local_int_t inew = opt->perm0[i];
    double d = dv[i];
    opt->diag[inew] = d;
    opt->idiag[inew] = 1.0/d;
  }
}

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints

  //EF////EF Test
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  //EF//if (mpi_rank == 0) matfile.open("matrix.dat");

#ifdef FTRACE
  ftrace_region_begin("vcycle_GenerateOPT");
#endif
  vcycle(GenerateOPT_STRUCT, A);
#ifdef FTRACE
  ftrace_region_end("vcycle_GenerateOPT");
#endif

  //EF Test
  //EF//vcycle(Optimize_CheckDone, A);

  // Set the pointers of diagonal elements
  vcycle(SetDiagonalPointer, A);
 
  // Set the permuted f2cOperator
  vcycle(SetF2cOperator, A);
  
  // Set communication table corresponding to the permuted matrix
  vcycle(SetCommTable, A);

  // Set the permuted vector b
  OPT*    opt    = (OPT*)(A.optimizationData);
  Vector* perm_b = new Vector;
  InitializeVector(*perm_b, b.localLength);
  PermVector(opt->iperm0, b, *perm_b);    // perm_b <-- P^(-1) b
  CopyVector(*perm_b, b);    // perm_b <-- P^(-1) b
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  // Size of the following 2 arrays!
  // ell->a     = new double[lda*m];        // value
  // ell->ja    = new local_int_t[lda*m];   // column index
  int numberOfMgLevels = 4; // Number of levels including first
  const SparseMatrix * curLevelMatrix = &A;
  double sizea, sizeja, retval;
  retval = 0.0;
  for(int level = 0; level< numberOfMgLevels; ++level){
    int lda = ((OPT*)((*curLevelMatrix).optimizationData))->ell->lda;
    int m   = ((OPT*)((*curLevelMatrix).optimizationData))->ell->m;
//  fprintf(stderr, "### level %d\n", level);
//  fprintf(stderr, "lda: %d\n", lda);
//  fprintf(stderr, "m:   %d\n", m);
    double sizea  = ((double) lda)*((double) m)*((double) sizeof(double));
    double sizeja = ((double) lda)*((double) m)*((double) sizeof(local_int_t));
    retval += sizea + sizeja;
    // add diag, idiag, the 2 perm vectors
    double size_diag = (double)lda * sizeof(double);
    double size_perm = (double)lda * sizeof(local_int_t);
    retval += size_diag * 2 + size_perm * 2;
    // add space for work1, work2
    retval += size_diag * 2;
    // TODO: add halo matrix info, if proper matrix format
    curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
  }
//fprintf(stderr, "Memory Size[MB]: %e\n", retval/1.0e9);
  retval *= ((double) A.geom->size);
//fprintf(stderr, "Memory Size[MB]: %e\n", retval/1.0e9);
  return retval;
}


//#########################################################
// Driver for matrix optimization
//#########################################################
int GenerateOPT_STRUCT(SparseMatrix & A)
{
  OPT        *opt = new OPT;
  ELL        *ell = new ELL;
  HALO      *halo = new HALO;
  opt->ell        = ell;
  local_int_t n   = ell->n        = A.localNumberOfRows;
  opt->halo       = halo;

  // EF: TryFreeMemory
  delete [] A.mtxIndG[0];
  delete [] A.mtxIndG; A.mtxIndG = 0;

  A.optimizationData = opt;

  if (n <= 0) return(0);

  opt->diag   = new double[n*4];  // allocate multiple arrays in one step
  opt->idiag  = opt->diag + n;
  opt->work1  = opt->idiag + n;
  opt->work2  = opt->work1 + n;
  local_int_t *icolor = opt->icolor = new local_int_t[n];  // color of each row

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
  //  Make the permutation matrix for Level scheduling or Multicolor ordering  //
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

  // Find the maximum degree of each node
  local_int_t maxNonzerosInRow = 0;
  for(local_int_t i=0; i<n; i++) {
    if (A.nonzerosInRow[i] > maxNonzerosInRow)
      maxNonzerosInRow = A.nonzerosInRow[i];
  }
  ell->m = maxNonzerosInRow - 1;

  Optimize_Hyperplane(A);

  //##########################################################

  // Search the maximum number of colors
  local_int_t maxcolor = 0, numcolors = 0;
  for(local_int_t i=0; i<n; i++) {
    if (icolor[i] > maxcolor)
        maxcolor = icolor[i];
  }
  maxcolor++;

#ifdef HPCG_DEBUG
  HPCG_fout     << "maxcolor = " << maxcolor << std::endl;
#endif

  ////////////////////////////////////////////
  // Create permutations according to coloring
  ////////////////////////////////////////////
  
  opt->perm0   = new local_int_t[n+n+maxcolor+1]; // multiple allocs in one
  opt->iperm0  = opt->perm0 + n;
  opt->icptr   = opt->iperm0 + n;
    
#ifdef VHCALLxx
  vhcallVE_GetPerm(n, maxcolor, icolor, opt->perm0, opt->icptr);
#else

  local_int_t icacc[maxcolor][VLEN];
  for (local_int_t i=0; i<maxcolor+1; i++)
    opt->icptr[i]=0;

  //for (local_int_t i=0; i<n; i++)
  //  ++opt->icptr[icolor[i]+1]; // Count up for each color

  for (int i = 0; i < maxcolor * VLEN; i++)
    icacc[0][i] = 0;

  for (local_int_t i_=0; i_<n; i_+=VLEN) {
    if (i_ == 0) {
      #pragma _NEC shortloop
      #pragma _NEC ivdep
      for (int i = i_; i < MIN(i_ + VLEN, n); i++)
        icacc[icolor[i]][i - i_] = 1;
    } else {
      #pragma _NEC shortloop
      #pragma _NEC ivdep
      for (int i = i_; i < MIN(i_ + VLEN, n); i++)
        ++icacc[icolor[i]][i - i_];
    }
  }
  
  for (int i = 0; i < VLEN; i++)
    for (local_int_t ic=0; ic<maxcolor; ic++)
      opt->icptr[ic+1] += icacc[ic][i];

  for (local_int_t ic=1; ic<maxcolor+1; ic++)
    opt->icptr[ic] += opt->icptr[ic-1];

  //
  //  Make the permutation matrices perm0[] and iperm0[]
  //
  local_int_t itmp[maxcolor];
  for (local_int_t ic=0; ic<maxcolor; ic++)
    itmp[ic] = opt->icptr[ic];
  for (local_int_t i=0; i<n; i++) {
    local_int_t ic = icolor[i];
    opt->perm0[i]= itmp[ic];              // 0 origin
    itmp[ic]++;
  }

#endif // ! VHCALL

#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i<n; i++) {
    opt->iperm0[opt->perm0[i]]=i;        // 0 origin
  }

  opt->maxcolor  = maxcolor;

  const char* env_p = std::getenv("HPCG_ASYOPT");
  if (env_p != nullptr && strcmp(env_p, "YES") == 0) {
    ell->a = nullptr;
  } else {
      Optimize_Store_ELL_L_U_halo(A);
  }

  // EF: try free some matrix stuff
  //-------------------------------
  delete [] A.matrixValues[0];
  delete [] A.matrixValues; A.matrixValues = 0;
  delete [] A.mtxIndL[0];
  delete [] A.mtxIndL; A.mtxIndL = 0;

  //-------------------------------
  
#ifdef HPCG_DEBUG_MATRIX
  if (first_call) {
    auto file = std::fstream("matrix.bin", std::ios::out | std::ios::binary);
    file.write((char *)&n, sizeof(local_int_t));
    file.write((char *)&m, sizeof(local_int_t));
    file.write((char *)&ell->ja[0], sizeof(local_int_t)*n*m);
    file.write((char *)&maxcolor, sizeof(int));
    file.write((char *)&opt->icptr[0], sizeof(int)*(maxcolor+1));
    file.close();
    first_call = 0;
  }
  if(n<100) {
    cout<<"permuted matrix indices ell->ja\n";
    for(int i=0; i<n;i++) {
      cout<<fixed<<setw(5)<<i;
      for(int j=0; j<m;j++)
        cout<<fixed<<setw(5)<<ell->ja[i+lda*j];
      cout<<"\n"<<flush;
    }
  }
#endif
  return 0;
}

int SetDiagonalPointer(SparseMatrix & A)
{
  OPT* opt = (OPT*)(A.optimizationData);
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for(local_int_t i=0; i<A.localNumberOfRows; i++){
    local_int_t inew = opt->perm0[i];
    A.matrixDiagonal[i] = &opt->diag[inew];
  }
  return(0);
}

int SetF2cOperator(SparseMatrix & A) {
  OPT* optf = (OPT*)(A.optimizationData);
  if(A.Ac){
    OPT* optc = (OPT*)(A.Ac->optimizationData);
    local_int_t* f2c = new local_int_t[A.mgData->rc->localLength];

    //call sblas routine
    //ssctr_( &A.mgData->rc->localLength, A.mgData->f2cOperator, optc->perm, f2c);  // 1 origin
    for (local_int_t i = 0; i < A.mgData->rc->localLength; i++)
      f2c[optc->perm0[i]] = A.mgData->f2cOperator[i];
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
    for (local_int_t i=0; i<A.mgData->rc->localLength; i++)
      A.mgData->f2cOperator[i] = optf->perm0[f2c[i]];
    delete [] f2c;
  }
  return 0;
}

int SetCommTable(SparseMatrix & A) {
#ifndef HPCG_NO_MPI
  OPT* opt = (OPT*)(A.optimizationData);
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * iw = new local_int_t[A.totalToBeSent];
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i<totalToBeSent; i++) iw[i] = A.elementsToSend[i];
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i<totalToBeSent; i++) A.elementsToSend[i]  = opt->perm0[iw[i]];
  delete [] iw;
#endif
  return 0;
}

int PermVector(const local_int_t * p, const Vector & x, Vector & px) {
  local_int_t localLength = x.localLength;
  // call sblas routine
  //dgthr_( &localLength, x.values, px.values, p);
  for (local_int_t i = 0; i < localLength; i++)
    px.values[i] = x.values[p[i]];
  return 0;
}
