#include <sim.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#define GOOD_BLOCK_SIZE 64

__device__
static real_v force_part(const double& q0[3], const double& q1[3], double c0, double c1)
{
  double diff[3];
  double dist = 0.0;
  for(int i = 0; i < 3; i++)
  {
    diff[i] = q1[i] - q0[i];
    double dist += diff[i] * diff[i];
  }
  //Deal with edge issues
  if(dist == 0.0)
  {
    dist = 1e-300;
    for(int i = 0; i < 3; i++) diff[i] = 1e-10;
  }
  dist = sqrt(dist);
  double mag_F = -c0 * c1 /dist;
  for(int i = 0; i < 3; i++) diff[i] = diff[i] * mag_F/dist;
  return real_v{diff[0], diff[1], diff[2]};
}

__global__
static void compute_forces(real_v* q0, real_v* c, real_v* out, uint64 N)
{
  int id = blockIdx.x * blockDim.x + blockIdx.x;
  //out is an array of force components, for each particle there are N forces, for N particles
  int big = floor((sqrt(8*id +1)-1)/2);
  int lil = id - y*(y+1)/2;
  int q0i = big;
  int q1i = lil;
  if(q1i == q0i) q1i = N;
  real_v to_out = force_part(q0[q0i].val, q0[q1i].val, c[q0i], c[q1i]);
  out[big*N + lil] = to_out;
  for(int i = 0; i < 3; i++) to_out.vals[i] = -to_out.vals[i];
  out[lil*N + big] = to_out;
}

__global__
static void q1_update(real_v* q0, real_v* qdot0, double dt, uint64 N)
{
  id = blockIdx.x * blockDim.x + blockIdx.x;
  if(N < dim)
  {
    q0[id].vals[blockIdx.y] += dt * qdot0[id].vals[blockIdx.y];
  }
}

__global__
static void qdot1_update(real_v* qdot0, real_v* Fq1, double dt, uint64 N, double* m)
{
  id = blockIdx.x * blockDim.x + blockIdx.x;
  if(N < dim)
  {
    q0[id].vals[blockIdx.y] += dt * Fq1[id].vals[blockIdx.y]/m[dim];
  }
}

static struct comp_Fq1
{
  comp_Fq1() {}
  real_v operator() (real_v a, real_v b)
  {
    real_v ret;
    for(int i = 0; i < 3; i++) ret[i] = a[i] + b[i];
    return ret;
  }
};

real_v* out;
real_v* Fq1;

thrust::equal_to<int> pred;
comp_Fq1 spec_add;

static real_v* step(real_v* q, real_v* qdot, double* m, double* c, double QE, double r, double dt, uint64 N)
{
  int q1_blocks = (N % GOOD_BLOCK_SIZE) ? (N/GOOD_BLOCK_SIZE) + 1 : N/GOOD_BLOCK_SIZE;
  dim3 q1_thread(GOOD_BLOCK_SIZE, 3);
  //Update q1
  q1_update<<<q1_blocks, q1_thread>>>(q, qdot);
  auto counter = thrust:make_counting_iterator(0);
  auto philled = repeat_iterator<thrust::counting_iterator<int>>(counter, N);
  //Update qdot1
  //Compute forces
  int blah = N*(N+1) /2;
  int forces_blocks = (blah % GOOD_BLOCK_SIZE) ? (blah / GOOD_BLOCK_SIZE) + 1 : N / GOOD_BLOCK_SIZE;
  int forces_thread = GOOD_BLOCK_SIZE;
  cudaDeviceSynchronize();
  compute_forces<<<forces_blocks, forces_thread>>>(q, c, out, N);
  cudaDeviceSynchronize();
  thrust::reduce_by_key(thrust::device, philled, philled + N * N, out, thrust::make_discard_iterator(), Fq1, pred, spec_add);
  cudaDeviceSynchronize();
  qdot1_update<<<q1_blocks, q1_thread>>>(qdot, Fq1, dt, N, m);
  cudaDeviceSynchronize();

}

output* n_body_eval(real_v* p, real_v* v, double* m, double* q, double QE, double r, double dt, uint64 n, uint64 N)
{

}
