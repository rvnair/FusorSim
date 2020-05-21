#include <sim.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <ofstream>

constexpr GOOD_BLOCK_SIZE = 64;

__device__
static real_v force_part(const double& q0[REAL_V_DIM], const double& q1[REAL_V_DIM], double c0, double c1)
{
  double diff[REAL_V_DIM];
  double dist = 0.0;
  for(int i = 0; i < REAL_V_DIM; i++)
  {
    diff[i] = q1[i] - q0[i];
    double dist += diff[i] * diff[i];
  }
  //Deal with edge issues
  if(dist == 0.0)
  {
    dist = 1e-300;
    for(int i = 0; i < REAL_V_DIM; i++) diff[i] = 1e-10;
  }
  dist = sqrt(dist);
  double mag_F = -c0 * c1 /dist;
  real_v ret;
  for(int i = 0; i < REAL_V_DIM; i++) ret.vals[i] = diff[i] * mag_F/dist;
  return ret;
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
  if(q1i < N)
  {
    for(int i = 0; i < REAL_V_DIM; i++) out[lil*N + big].vals[i] = -to_out.vals[i];
  }
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
    for(int i = 0; i < REAL_V_DIM; i++) ret[i] = a[i] + b[i];
    return ret;
  }
};

thrust::equal_to<int> pred;
comp_Fq1 spec_add;

cudaStream_t stream1, stream2;

constexpr dim3 q1_thread(GOOD_BLOCK_SIZE, REAL_V_DIM);
constexpr int forces_thread = GOOD_BLOCK_SIZE;
constexpr int streams = 6;

static void step(real_v* q, real_v* qdot, double* m, double* c, double dt, uint64 N, real_v* out, real_v* Fq1, real_v* temp_q, const cudaStream_t& s[streams], void (*passed)(output), output prepped)
{
  cudaEvent_t event;
  cudaEventCreate(&event);
  int q1_blocks = (N % GOOD_BLOCK_SIZE) ? (N/GOOD_BLOCK_SIZE) + 1 : N/GOOD_BLOCK_SIZE;
  //Update q1
  q1_update<<<q1_blocks, q1_thread,0,s1>>>(q, qdot, dt, N);
  cudaEventRecord(event, s[0]);
  cudaStreamWaitEvent(s[1], event);
  cudaMemcpyAsync(temp_q, q, sizeof(real_v)*N, cudaMemcpyDeviceToHost, s[1]);
  //Update qdot1
  //Compute forces
  int blah = N*(N+1) /2;
  int forces_blocks = (blah % GOOD_BLOCK_SIZE) ? (blah / GOOD_BLOCK_SIZE) + 1 : N / GOOD_BLOCK_SIZE;
  compute_forces<<<forces_blocks, forces_thread,0,s[0]>>>(q, c, out, N);
  auto counter = thrust:make_counting_iterator(0);
  auto philled = repeat_iterator<thrust::counting_iterator<int>>(counter, N);
  thrust::reduce_by_key(thrust::cuda::par.on(s[0]), philled, philled + N * N, out, thrust::make_discard_iterator(), Fq1, pred, spec_add);
  qdot1_update<<<q1_blocks, q1_thread,0,s[0]>>>(qdot, Fq1, dt, N, m);
  cudaStreamSynchronize(s[1]);
  prepped.part = temp_q;
  passed(prepped);
  cudaDeviceSynchronize();
}

output* n_body_eval(real_v* q, real_v* v, double* m, double* c, double QE, double r, double dt, uint64 n, uint64 N, void (*passed)(output))
{
  real_v* dev_q;
  real_v* dev_v;
  double* dev_m;
  double* dev_c;
  real_v* d_out;
  real_v* d_Fq1;
  cudaStream_t s[streams];
  for(int i = 0; i < streams; i++) cudaStreamCreate(&s[i]);
  output prep;
  //For optimization place the central charge in the same array
  cudaMalloc(dev_q, sizeof(real_v)*(N+1));
  cudaMalloc(dev_v, sizeof(real_v)*N);
  cudaMalloc(dev_m, sizeof(double)*N);
  cudaMalloc(dev_c, sizeof(double)*(N+1));
  cudaMalloc(d_out, sizeof(real_v)*N*N);
  cudaMalloc(d_Fq1, sizeof(real_v)*N);
  cudaDeviceSynchronize();
  cudaMemcpyAsync(dev_q, q, sizeof(real_v)*N, cudaMemcpyHostToDevice, s[0]);
  cudaMemcpyAsync(dev_v, v, sizeof(real_v)*N, cudaMemcpyHostToDevice, s[1]);
  cudaMemcpyAsync(dev_m, m, sizeof(double)*N, cudaMemcpyHostToDevice, s[2]);
  cudaMemcpyAsync(dev_c, c, sizeof(double)*N, cudaMemcpyHostToDevice, s[3]);
  cudaMemcpyAsync(&dev_c[N],&QE,sizeof(double),cudaMemcpyHostToDevice,s[4]);
  cudaMemsetAsync(&dev_q[N],0,sizeof(real_v), cudaMemcpyHostToDevice, s[5]);
  cudaDeviceSynchronize();
  out.numb = N;
  out.part = nullptr;
  for(uint64 i = 0; i < n; i++)
  {
    out.step = i;
    step(dev_q, dev_v, dev_m, dev_c, dt, N, d_out, d_Fq1, q, s, passed);
  }
  cudaFree(dev_q);
  cudaFree(dev_v);
  cudaFree(dev_m);
  cudaFree(dev_c);
  cudaFree(d_out);
  cudaFree(d_Fq1);
  cudaDeviceSynchronize();

}
