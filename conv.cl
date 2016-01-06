#define Min(a,b) ((a)<(b)?(a):(b))


// First naive implementation
__kernel void Conv(const int out_0, const int out_1, const int out_2,
                      const int d_0, const int d_1, const int d_2,
                      const int kernel_0, const int kernel_1,
                      const int kernel_stride0,const int kernel_stride1,
                      const __global __attribute__ ((aligned (64))) float* wmatPtr ,
                      const __global  float* dataP ,
                      __global float* resPtr ,
                      __local float* dataCache ,
                      __local float* wmatCache ,
                      __local float* resCache ) {

#define Opti12

#ifdef BASE
    const int ii = get_group_id(0);
    const int jj = get_local_id(0);

    for(int kk=0;kk<out_2;kk++) {
      float tempAcc = 0.0;
      for(int mm=0;mm<d_0;mm++)
        for(int pp=0;pp<kernel_0;pp++)
          for(int tt=0;tt<kernel_1;tt++)
            //resPtr[ii][jj][kk]+=wmatPtr[ii][mm][pp][tt]*dataPtr[mm][kernel_stride0*jj+pp][kernel_stride1*kk+tt];
            tempAcc += wmatPtr[ii*d_0*kernel_0*kernel_1+mm*kernel_0*kernel_1+pp*kernel_1+tt]* \
            dataP[mm*d_1*d_2+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
      resPtr[ii*out_1*out_2+jj*out_2+kk] = tempAcc;
    }
#endif

#ifdef Opti0
    const int ii = get_group_id(0);
    const int jj = get_local_id(0);
    for(int kk=0;kk<out_2;kk++)
      resPtr[ii*out_1*out_2+jj*out_2+kk] = 0.0;

    for(int mm=0;mm<d_0;mm++)
      for(int kk=0;kk<out_2;kk++)
        for(int pp=0;pp<kernel_0;pp++)
          for(int tt=0;tt<kernel_1;tt++)
            resPtr[ii*out_1*out_2+jj*out_2+kk] += wmatPtr[ii*d_0*kernel_0*kernel_1+mm*kernel_0*kernel_1+pp*kernel_1+tt]* \
              dataP[mm*d_1*d_2+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];

#endif


#ifdef Opti1
const int ii = get_group_id(0);
const int jj = get_local_id(0);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;

for(int kk=0;kk<out_2;kk++)
    resCache[jj*out_2+kk] = 0.0;
barrier( CLK_LOCAL_MEM_FENCE );

for(int mm=0;mm<d_0;mm++)
  for(int kk=0;kk<out_2;kk++)
    for(int pp=0;pp<kernel_0;pp++){
      #pragma unroll
      for(int tt=0;tt<kernel_1;tt++)
        resCache[jj*out_2+kk] += wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1+tt]* \
                dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
    }

barrier( CLK_LOCAL_MEM_FENCE );
for(int kk=0;kk<out_2;kk++)
  resPtr[ii*out_12+jj*out_2+kk] = resCache[jj*out_2+kk];

#endif


#ifdef Opti2
const int ii = get_group_id(0);
const int jj = get_local_id(0);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;

const int stride0 = 1,stride1=1;
const int pad0=1, pad1=1;

for(int kk=0;kk<out_2;kk++)
    resCache[jj*out_2+kk] = 0.0;
barrier( CLK_LOCAL_MEM_FENCE );

for(int mm=0;mm<d_0;mm++)
  for(int kk=0;kk<out_2;kk++) {
    if(jj==0) {
      for(int i1=0;i1<kernel_0;i1++)
        for(int j1=0;j1<kernel_1;j1++)
          //wmatCache[i1*kernel_1+j1] = wmatPtr[ii*d_0*kernel_0*kernel_1+mm*kernel_0*kernel_1+i1*kernel_1+j1];
          wmatCache[i1*kernel_1+j1] = wmatPtr[ii*dkernel_01+mm*kernel_01+i1*kernel_1+j1];
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    for(int pp=0;pp<kernel_0;pp++)
      for(int tt=0;tt<kernel_1;tt++)
        resCache[jj*out_2+kk] += wmatCache[pp*kernel_1+tt]*dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
        //resCache[jj*out_2+kk] += wmatCache[pp*kernel_1+tt]*dataCache[(jj*stride0+pp)*d_2+kk*stride1+tt];
  }

barrier( CLK_LOCAL_MEM_FENCE );
for(int kk=0;kk<out_2;kk++)
  resPtr[ii*out_12+jj*out_2+kk] = resCache[jj*out_2+kk];

#endif

#ifdef Opti3   //based on Opti1
const int ii = get_group_id(0);
const int jj = get_local_id(0);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;

float4 init0;
init0.x=0.0;  init0.y=0.0;  init0.y=0.0;  init0.w=0.0;
//float4 tmpb = (*((__global float4*)&B[indexB]));
//float4 tmpa = (*((__local float4*)&dataCacheA[indexA]));

for(int kk=0;kk<out_2;kk+=4)
    (*((__local float4*)&resCache[jj*out_2+kk])) = init0;
barrier( CLK_LOCAL_MEM_FENCE );

for(int mm=0;mm<d_0;mm++)
  for(int kk=0;kk<out_2;kk++) {
    //for(int pp=0;pp<kernel_0;pp++) {
      //for(int tt=0;tt<kernel_1;tt++)
        //resCache[jj*out_2+kk] += wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1+tt]* \
                dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
      float sum1=0.0;
      //float3 tempw = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1]));
      //float3 tempd = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk]));
      float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1]));
      float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.x*tempd1.x;
      sum1 += tempw1.y*tempd1.y;
      sum1 += tempw1.z*tempd1.z;

      float sum2=0.0;
      float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1]));
      float3 tempd2 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw2.x*tempd2.x;
      sum2 += tempw2.y*tempd2.y;
      sum2 += tempw2.z*tempd2.z;

      float sum3=0.0;
      float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
      float3 tempd3 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw3.x*tempd3.x;
      sum3 += tempw3.y*tempd3.y;
      sum3 += tempw3.z*tempd3.z;

      resCache[jj*out_2+kk] += sum1+sum2+sum3;
    }

barrier( CLK_LOCAL_MEM_FENCE );
for(int kk=0;kk<out_2;kk+=4)
  //resPtr[ii*out_12+jj*out_2+kk] = resCache[jj*out_2+kk];
  (*((__global float4*)&resPtr[ii*out_12+jj*out_2+kk])) = (*((__local float4*)&resCache[jj*out_2+kk]));

#endif


#ifdef Opti4   //change the local size and global size
const int ii = get_group_id(0);
const int jj = get_local_id(0);
const int kk = get_local_id(1);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;

float4 init0;
init0.x=0.0;  init0.y=0.0;  init0.y=0.0;  init0.w=0.0;
//float4 tmpb = (*((__global float4*)&B[indexB]));
//float4 tmpa = (*((__local float4*)&dataCacheA[indexA]));

//for(int kk=0;kk<out_2;kk+=4)
    //(*((__local float4*)&resCache[jj*out_2+kk])) = init0;
//resCache[jj*out_2+kk] = 0.0;
//barrier( CLK_LOCAL_MEM_FENCE );
float sum = 0.0;

for(int mm=0;mm<d_0;mm++){
  //for(int pp=0;pp<kernel_0;pp++) {
      //for(int tt=0;tt<kernel_1;tt++)
        //resCache[jj*out_2+kk] += wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1+tt]* \
                dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
      float sum1=0.0;
      //float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1]));
      //float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk]));
      float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01]));
      float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.x*tempd1.x;
      sum1 += tempw1.y*tempd1.y;
      sum1 += tempw1.z*tempd1.z;
      //resCache[jj*out_2+kk] += sum1;
      sum += sum1;

      float sum2=0.0;
      float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+kernel_1]));
      float3 tempd2 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw2.x*tempd2.x;
      sum2 += tempw2.y*tempd2.y;
      sum2 += tempw2.z*tempd2.z;
      //resCache[jj*out_2+kk] += sum2;
      sum += sum2;

      float sum3=0.0;
      float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
      float3 tempd3 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw3.x*tempd3.x;
      sum3 += tempw3.y*tempd3.y;
      sum3 += tempw3.z*tempd3.z;
      //resCache[jj*out_2+kk] += sum3;
      sum += sum3;
  }

//barrier( CLK_LOCAL_MEM_FENCE );
//for(int kk=0;kk<out_2;kk+=4)
  //(*((__global float4*)&resPtr[ii*out_12+jj*out_2+kk])) = (*((__local float4*)&resCache[jj*out_2+kk]));
//resPtr[ii*out_12+jj*out_2+kk] = resCache[jj*out_2+kk];
resPtr[ii*out_12+jj*out_2+kk] = sum;
#endif


#ifdef Opti5  //reorrgansed the local size and global size
const int index = get_global_id(0);
  const int ii = index/(out_1*out_2);
  const int jj = (index%(out_1*out_2))/out_2;
  const int kk = (index%(out_1*out_2))%out_2;

  const int out_12 = out_1*out_2;
  const int kernel_01 = kernel_0*kernel_1;
  const int dkernel_01 = d_0*kernel_01;
  const int d_12 = d_1*d_2;

  //resCache[jj*out_2+kk] = 0.0;
  //barrier( CLK_LOCAL_MEM_FENCE );
  int index1 = ii*out_12+jj*out_2+kk;
  //resPtr[ii*out_12+jj*out_2+kk] = 0.0;
  resPtr[index1] = 0.0;
  int index2 = ii*dkernel_01;
  int index3 = kernel_stride0*jj*d_2+kernel_stride1*kk;
  int index4 = index2;
  int index5 = index3;
  int index6=0;
  int index7=0;

  for(int mm=0;mm<d_0;mm++){
    index2 = index4;
    index3 = index5;
    for(int pp=0;pp<kernel_0;pp++) {
        float sum=0.0;
        //float3 tempw = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1]));
        //float3 tempd = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk]));
        //float3 tempw = (*((__global float3*)&wmatPtr[index2+mm*kernel_01+pp*kernel_1]));
        //float3 tempd = (*((__global float3*)&dataP[mm*d_12+index3+pp*d_2]));
        //float3 tempw = (*((__global float3*)&wmatPtr[index2+mm*kernel_01]));
        //float3 tempd = (*((__global float3*)&dataP[mm*d_12+index3]));
        float3 tempw = (*((__global float3*)&wmatPtr[index2+index6]));
        float3 tempd = (*((__global float3*)&dataP[index3+index7]));
        index2+=kernel_1;
        index3+=d_2;
        sum += tempw.x*tempd.x;
        sum += tempw.y*tempd.y;
        sum += tempw.z*tempd.z;
        //resCache[jj*out_2+kk] += sum;
        //resPtr[ii*out_12+jj*out_2+kk] += sum;
        resPtr[index1] += sum;
    }
    index6+=kernel_01;
    index7+=d_12;
  }


  //barrier( CLK_LOCAL_MEM_FENCE );
  //resPtr[ii*out_12+jj*out_2+kk] = resCache[jj*out_2+kk];
#endif



#ifdef Opti6  //reorrgansed the local size and global size
const int index = get_global_id(0);
  const int ii = index/(out_1*out_2);
  const int jj = (index%(out_1*out_2))/out_2;
  const int kk = (index%(out_1*out_2))%out_2;

  const int out_12 = out_1*out_2;
  const int kernel_01 = kernel_0*kernel_1;
  const int dkernel_01 = d_0*kernel_01;
  const int d_12 = d_1*d_2;

  //resCache[jj*out_2+kk] = 0.0;
  //barrier( CLK_LOCAL_MEM_FENCE );
  resPtr[ii*out_12+jj*out_2+kk] = 0.0;

  for(int mm=0;mm<d_0;mm+=4)
    for(int pp=0;pp<kernel_0;pp++)
      for(int tt=0;tt<kernel_1;tt++){
        float sum=0.0;
        float4 tempw;
        tempw.x = (*((__global float*)&wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1+tt]));
        tempw.y = (*((__global float*)&wmatPtr[ii*dkernel_01+(mm+1)*kernel_01+pp*kernel_1+tt]));
        tempw.z = (*((__global float*)&wmatPtr[ii*dkernel_01+(mm+2)*kernel_01+pp*kernel_1+tt]));
        tempw.w = (*((__global float*)&wmatPtr[ii*dkernel_01+(mm+3)*kernel_01+pp*kernel_1+tt]));
        float4 tempd;
        tempd.x = dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
        tempd.y = dataP[(mm+1)*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
        tempd.z = dataP[(mm+2)*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
        tempd.w = dataP[(mm+3)*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];

        sum += tempw.x*tempd.x;
        sum += tempw.y*tempd.y;
        sum += tempw.z*tempd.z;
        sum += tempw.w*tempd.w;
        resPtr[ii*out_12+jj*out_2+kk] += sum;
    }


#endif


#ifdef Opti7   //based on Opti5 unroll kernel for
const int index = get_global_id(0);
  const int ii = index/(out_1*out_2);
  const int jj = (index%(out_1*out_2))/out_2;
  const int kk = (index%(out_1*out_2))%out_2;

  const int out_12 = out_1*out_2;
  const int kernel_01 = kernel_0*kernel_1;
  const int dkernel_01 = d_0*kernel_01;
  const int d_12 = d_1*d_2;

  int index1 = ii*out_12+jj*out_2+kk;
  resPtr[index1] = 0.0;
  int index2 = ii*dkernel_01;
  int index3 = kernel_stride0*jj*d_2+kernel_stride1*kk;
  int index4 = index2;
  int index5 = index3;
  int index6=0;
  int index7=0;

  float3 sum0;
  sum0.x=0.0;
  sum0.y=0.0;
  sum0.z=0.0;
  float sum=0.0;


  for(int mm=0;mm<d_0;mm++){
    //for(int pp=0;pp<kernel_0;pp++) {
        //sum = sum0;
        //float3 sum1=sum0;
        float sum1=0.0;
        //float3 tempw = (*((__global float3*)&wmatPtr[index2+mm*kernel_01+pp*kernel_1]));
        //float3 tempd = (*((__global float3*)&dataP[mm*d_12+index3+pp*d_2]));
        float3 tempw1 = (*((__global float3*)&wmatPtr[index2+index6]));
        float3 tempd1 = (*((__global float3*)&dataP[index3+index7]));
        sum1 += tempw1.x*tempd1.x;
        sum1 += tempw1.y*tempd1.y;
        sum1 += tempw1.z*tempd1.z;

        //float3 sum2=sum0;
        float sum2=0.0;
        float3 tempw2 = (*((__global float3*)&wmatPtr[index2+index6+kernel_1]));
        float3 tempd2 = (*((__global float3*)&dataP[index3+index7+d_2]));
        sum2 += tempw2.x*tempd2.x;
        sum2 += tempw2.y*tempd2.y;
        sum2 += tempw2.z*tempd2.z;

        //float3 sum3=sum0;
        float sum3=0.0;
        float3 tempw3 = (*((__global float3*)&wmatPtr[index2+index6+2*kernel_1]));
        float3 tempd3 = (*((__global float3*)&dataP[index3+index7+2*d_2]));
        sum3 += tempw3.x*tempd3.x;
        sum3 += tempw3.y*tempd3.y;
        sum3 += tempw3.z*tempd3.z;
        //resCache[jj*out_2+kk] += sum;
        //resPtr[ii*out_12+jj*out_2+kk] += sum;
        //sum.x = sum1.x+sum2.x+sum3.x;
        //sum.y = sum1.y+sum2.y+sum3.y;
        //sum.z = sum1.z+sum2.z+sum3.z;
        //resPtr[index1] += sum1+sum2+sum3;
        sum += sum1+sum2+sum3;

    //}
    index6+=kernel_01;
    index7+=d_12;
  }

  resPtr[index1] = sum;

#endif

#ifdef Opti8   //change the local size {4,4} and global size
const int ii = get_group_id(0);
//const int jj = get_local_id(0);
//const int kk = get_local_id(1);
const int j = get_local_id(0);
const int k = get_local_id(1);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;

float3 init0;
init0.x=0.0;  init0.y=0.0;  init0.y=0.0;  //init0.w=0.0;
//float4 tmpb = (*((__global float4*)&B[indexB]));
//float4 tmpa = (*((__local float4*)&dataCacheA[indexA]));

//for(int jj=j*3;jj<j*3+3;jj++)
//  for(int kk=k*3;kk<k*3+3;kk++)
//    resPtr[ii*out_12+jj*out_2+kk] = 0.0;

//for(int kk=0;kk<out_2;kk+=4)
//    (*((__local float4*)&resCache[jj*out_2+kk])) = init0;
//resCache[jj*out_2+kk] = 0.0;
//barrier( CLK_LOCAL_MEM_FENCE );
//float sum = 0.0;
for(int jj=j*3;jj<j*3+3;jj++)
//  for(int kk=k*3;kk<k*3+3;kk++)
    (*((__local float3*)&resCache[jj*out_2+3*k])) = init0;
barrier( CLK_LOCAL_MEM_FENCE );

for(int mm=0;mm<d_0;mm++){
  /*for(int jj=j*3;jj<j*3+3;jj++)
    for(int kk=k*3;kk<k*3+3;kk++)
      (*((__local float4*)&resCache[jj*out_2+kk])) = init0;
  barrier( CLK_LOCAL_MEM_FENCE );*/
  //for(int pp=0;pp<kernel_0;pp++) {
      //for(int tt=0;tt<kernel_1;tt++)
        //resCache[jj*out_2+kk] += wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1+tt]* \
                dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];

  float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01]));
  float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+kernel_1]));
  float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
  for(int jj=j*3;jj<j*3+3;jj++)
    for(int kk=k*3;kk<k*3+3;kk++){
      float sum = 0.0;
      float sum1=0.0;
      //float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1]));
      //float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk]));
      //float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01]));
      float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.x*tempd1.x;
      sum1 += tempw1.y*tempd1.y;
      sum1 += tempw1.z*tempd1.z;
      //resCache[jj*out_2+kk] += sum1;
      sum += sum1;

      float sum2=0.0;
      //float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+kernel_1]));
      float3 tempd2 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw2.x*tempd2.x;
      sum2 += tempw2.y*tempd2.y;
      sum2 += tempw2.z*tempd2.z;
      //resCache[jj*out_2+kk] += sum2;
      sum += sum2;

      float sum3=0.0;
      //float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
      float3 tempd3 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw3.x*tempd3.x;
      sum3 += tempw3.y*tempd3.y;
      sum3 += tempw3.z*tempd3.z;
      //resCache[jj*out_2+kk] += sum3;
      sum += sum3;
      resCache[jj*out_2+kk] += sum;
      //resPtr[ii*out_12+jj*out_2+kk] += sum;
  }

/*barrier( CLK_LOCAL_MEM_FENCE );
//for(int kk=0;kk<out_2;kk+=4)
  //(*((__global float4*)&resPtr[ii*out_12+jj*out_2+kk])) = (*((__local float4*)&resCache[jj*out_2+kk]));
  for(int jj=j*3;jj<j*3+3;jj++)
    for(int kk=k*3;kk<k*3+3;kk++)
      resPtr[ii*out_12+jj*out_2+kk] += resCache[jj*out_2+kk];
//resPtr[ii*out_12+jj*out_2+kk] = sum;*/
}
for(int jj=j*3;jj<j*3+3;jj++)
//  for(int kk=k*3;kk<k*3+3;kk++)
    (*((__global float3*)&resPtr[ii*out_12+jj*out_2+3*k])) = (*((__local float3*)&resCache[jj*out_2+3*k]));
#endif


#ifdef Opti9   //based on Opti3
const int ii = get_group_id(0);
const int jj = get_local_id(0);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;

float4 init0;
init0.x=0.0;  init0.y=0.0;  init0.y=0.0;  init0.w=0.0;
//float4 tmpb = (*((__global float4*)&B[indexB]));
//float4 tmpa = (*((__local float4*)&dataCacheA[indexA]));

if(jj==1)
{
  for(int l1=1;l1<d_1-1;l1++){
    dataCache[l1*d_2] = 0.0;
    dataCache[l1*d_2+d_2-1] = 0.0;
  }

  for(int l2=0;l2<d_2;l2++){
    dataCache[l2] = 0.0;
    dataCache[(d_1-1)*d_2+l2] = 0.0;
  }
}


for(int kk=0;kk<out_2;kk+=4)
    (*((__local float4*)&resCache[jj*out_2+kk])) = init0;
barrier( CLK_LOCAL_MEM_FENCE );

for(int mm=0;mm<d_0;mm++) {
  float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1]));
  float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1]));
  float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));

  //float4 tempd1
  for(int kk=0;kk<out_2;kk++)
    //dataCache[jj+1][kk+1] = dataP[mm][jj][kk];
    dataCache[(jj+1)*d_2+kk+1] = dataP[mm*d_1*d_2+(jj+1)*d_2+kk+1];
  barrier( CLK_LOCAL_MEM_FENCE );


  for(int kk=0;kk<out_2;kk++) {
    //for(int pp=0;pp<kernel_0;pp++) {
      //for(int tt=0;tt<kernel_1;tt++)
        //resCache[jj*out_2+kk] += wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1+tt]* \
                dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
      float sum1=0.0;
      //float3 tempw = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1]));
      //float3 tempd = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk]));
      //float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1]));
      //float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      float3 tempd1 = (*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.x*tempd1.x;
      sum1 += tempw1.y*tempd1.y;
      sum1 += tempw1.z*tempd1.z;

      float sum2=0.0;
      //float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1]));
      //float3 tempd2 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      float3 tempd2 = (*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw2.x*tempd2.x;
      sum2 += tempw2.y*tempd2.y;
      sum2 += tempw2.z*tempd2.z;

      float sum3=0.0;
      //float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
      //float3 tempd3 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      float3 tempd3 = (*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw3.x*tempd3.x;
      sum3 += tempw3.y*tempd3.y;
      sum3 += tempw3.z*tempd3.z;

      resCache[jj*out_2+kk] += sum1+sum2+sum3;
    }
  }

barrier( CLK_LOCAL_MEM_FENCE );
for(int kk=0;kk<out_2;kk+=4)
  //resPtr[ii*out_12+jj*out_2+kk] = resCache[jj*out_2+kk];
  (*((__global float4*)&resPtr[ii*out_12+jj*out_2+kk])) = (*((__local float4*)&resCache[jj*out_2+kk]));

#endif

#ifdef Opti10
const int ii = get_group_id(0);
const int jj = get_local_id(0);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;

for(int kk=0;kk<out_2;kk++)
    resCache[jj*out_2+kk] = 0.0;
//barrier( CLK_LOCAL_MEM_FENCE );

//int ii=0;
//for(;ii<out_0;ii++){
  //for(int kk=0;kk<out_2;kk+=4)
  //    (*((__local float4*)&resCache[jj*out_2+kk])) = (0.0);

for(int mm=0;mm<d_0;mm++)
  /*for(int kk=0;kk<out_2;kk++)*/{
    //for(int pp=0;pp<kernel_0;pp++){
      //for(int tt=0;tt<kernel_1;tt++)
        float8 wmat;
        wmat = (*((__global float8*)&wmatPtr[ii*dkernel_01+mm*kernel_01]));
        float wmat8 = wmatPtr[ii*dkernel_01+mm*kernel_01+8];

        float4 resD1 = (0.0);
        float8 resD2 = (0.0);

        //float16 dataR0,dataR1,dataR2,dataR3,dataR4,dataR5,dataR6,dataR7,dataR8,dataR9,dataR10,dataR11,dataR12,dataR13;
        if(jj<d_1/2){
          (*((__local float16*)&dataCache[jj*2*d_2])) = (*((__global float16*)&dataP[mm*d_12+jj*2*d_2]));
          (*((__local float16*)&dataCache[jj*2*d_2+d_2])) = (*((__global float16*)&dataP[mm*d_12+jj*2*d_2+d_2]));
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        /////////////////////kk=0
        int kk=0;
        float sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];//sum1 += wmat.s0*dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

        float sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

        float sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD1.s0 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=1
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD1.s1 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=2
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD1.s2 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=3
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD1.s3 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;

        (*((__local float4*)&resCache[jj*out_2])) += resD1;


        /////////////////////kk=4
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD2.s0 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=5
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD2.s1 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=6
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD2.s2 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=7
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD2.s3 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=8
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD2.s4 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;

        /////////////////////kk=9
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD2.s5 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=10
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD2.s6 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        /////////////////////kk=11
        kk++;
         sum1=0.0,sum2=0.0,sum3=0.0;
        sum1 += wmat.s0*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmat.s1*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmat.s2*dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

         sum4=0.0,sum5=0.0,sum6=0.0;
        sum4 += wmat.s3*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
        sum5 += wmat.s4*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
        sum6 += wmat.s5*dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

         sum7=0.0,sum8=0.0,sum9=0.0;
        sum7 += wmat.s6*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
        sum8 += wmat.s7*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
        sum9 += wmat8*dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];
        //resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;
        resD2.s7 += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;


        (*((__local float8*)&resCache[jj*out_2+4])) += resD2;











        /*sum1 += wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1+0]* \
                dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+0];
        sum2 += wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1+1]* \
                dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+1];
        sum3 += wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1+2]* \
                dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk+2];

                float sum4=0.0,sum5=0.0,sum6=0.0;
                sum4 += wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1+0]* \
                        dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+0];
                sum5 += wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1+1]* \
                        dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+1];
                sum6 += wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1+2]* \
                        dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk+2];

                        float sum7=0.0,sum8=0.0,sum9=0.0;
                        sum7 += wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1+0]* \
                                dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+0];
                        sum8 += wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1+1]* \
                                dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+1];
                        sum9 += wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1+2]* \
                                dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk+2];



        resCache[jj*out_2+kk] += sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9;*/
    }
    barrier( CLK_LOCAL_MEM_FENCE );
    for(int kk=0;kk<out_2;kk+=4){
      (*((__global float4*)&resPtr[ii*out_12+jj*out_2+kk])) = (*((__local float4*)&resCache[jj*out_2+kk]));
      (*((__local float4*)&resCache[jj*out_2+kk])) = 0.0;
    }
    barrier( CLK_LOCAL_MEM_FENCE );
//  }



#endif

#ifdef Opti11   //change the local size and global size
const int i = get_group_id(0);
const int l_out_0 = out_0/get_num_groups(0);
//const int jj = get_local_id(0);
//const int kk = get_local_id(1);
const int j = get_local_id(0);
const int k = get_local_id(1);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;


int Ti = l_out_0;
int Tj = 1;
int Tk = 1;
int Tm = 1;


//for(int jj=j*3;jj<j*3+3;jj++)
//  for(int kk=k*3;kk<k*3+3;kk++)
//    (*((__local float3*)&resCache[jj*out_2+3*k])) = init0;
//barrier( CLK_LOCAL_MEM_FENCE );

for(int ii=i*l_out_0;ii<i*l_out_0+l_out_0;ii++)
  for(int jj=j*3;jj<j*3+3;jj++)
    for(int kk=k*3;kk<k*3+3;kk++)
      resPtr[ii*out_1*out_2+jj*out_2+kk] = 0.0;



//for(int ii=0;ii<out_0;ii++)
//  for(int jj=0;jj<out_1;jj++)
//    for(int kk=0;kk<out_2;kk++)
//      resPtr[ii*out_1*out_2+jj*out_2+kk] = 0.0;


for(int it=i*l_out_0;it<i*l_out_0+l_out_0;it+=Ti)
  for(int jt=j*3;jt<j*3+3;jt+=Tj)
    for(int kt=k*3;kt<k*3+3;kt+=Tk) {


      //for(int mt=0;mt<d_0;mt+=Tm)
      for(int mm=0;mm<d_0;mm+=Tm)
        for(int ii=it;ii<it+Ti;ii++)
          for(int jj=jt;jj<jt+Tj;jj++)
            for(int kk=kt;kk<kt+Tk;kk++) {
              float tempAcc = 0.0;
              //for(int mm=mt;mm<(((mt+Tm)<d_0)?:(mt+Tm),d_0);mm++)

              float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1]));
              float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1]));
              float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
              //for(int mm=mt;mm<(mt+Tm);mm++) {
                //for(int pp=0;pp<kernel_0;pp++){
                //for(int tt=0;tt<kernel_1;tt++)
                //resPtr[ii][jj][kk]+=wmatPtr[ii][mm][pp][tt]*dataPtr[mm][kernel_stride0*jj+pp][kernel_stride1*kk+tt];
                //tempAcc += wmatPtr[ii*d_0*kernel_0*kernel_1+mm*kernel_0*kernel_1+pp*kernel_1+tt]* \
                            dataP[mm*d_1*d_2+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
                float sum1=0.0,sum2=0.0,sum3=0.0;
                //float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1]));
                float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
                sum1 += tempw1.x*tempd1.x;
                sum1 += tempw1.y*tempd1.y;
                sum1 += tempw1.z*tempd1.z;

                //float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1]));
                float3 tempd2 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
                sum2 += tempw2.x*tempd2.x;
                sum2 += tempw2.y*tempd2.y;
                sum2 += tempw2.z*tempd2.z;

                //float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
                float3 tempd3 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
                sum3 += tempw3.x*tempd3.x;
                sum3 += tempw3.y*tempd3.y;
                sum3 += tempw3.z*tempd3.z;

                tempAcc += sum1+sum2+sum3;
              //}

              resPtr[ii*out_1*out_2+jj*out_2+kk] += tempAcc;
            }
    }


#endif



#ifdef Opti12
const int i = get_group_id(0);
const int j = get_local_id(0);
const int ii = get_global_id(0);
const int ls = get_local_size(0);

const int out_12 = out_1*out_2;
const int kernel_01 = kernel_0*kernel_1;
const int dkernel_01 = d_0*kernel_01;
const int d_12 = d_1*d_2;

/*__private float resCache1[144];

for(int jj=0;jj<out_1;jj++){
  //for(int kk=0;kk<out_2;kk+=4)
    //resCache[j*out_1*out_2+jj*out_2+kk] = 0.0;
    (*((__private float4*)&resCache1[jj*out_2])) = (float4)(0.0);
    (*((__private float8*)&resCache1[jj*out_2+4])) = (float8)(0.0);
  }*/
//barrier( CLK_LOCAL_MEM_FENCE );

for(int jj=0;jj<out_1;jj++)
  for(int kk=0;kk<out_2;kk+=4)
    (*((__global float4*)&resPtr[i*ls*out_12+j*out_12+jj*out_2+kk])) = (float4)(0.0);

for(int mm=0;mm<d_0;mm++){
  //float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1]));
  //float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1]));
  //float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
  float8 tempw1 = (*((__global float8*)&wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1]));
  float  tempwl = wmatPtr[ii*dkernel_01+mm*kernel_01+8];

  if(j<d_1){
    //for(int kk=0;kk<d_2;kk++)
      (*((__local float16*)&dataCache[j*d_2])) = (*((__global float16*)&dataP[mm*d_12+j*d_2]));
  }
  barrier( CLK_LOCAL_MEM_FENCE );
  __private float16 data1,data2,data3;
  data2 = (*((__local float16*)&dataCache[0]));
  data3 = (*((__local float16*)&dataCache[d_2]));
for(int jj=0;jj<out_1;jj++){
  data1 = data2;
  data2 = data3;
  data3 = (*((__local float16*)&dataCache[(jj+2)*d_2]));
  //for(int kk=0;kk<out_2;kk++){
    float sum1=0.0,sum2=0.0,sum3=0.0;

    //for(int pp=0;pp<kernel_0;pp++){
      //for(int tt=0;tt<kernel_1;tt++)

      //float3 tempw1 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+0*kernel_1]));
      /*float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.x*tempd1.x;
      sum1 += tempw1.y*tempd1.y;
      sum1 += tempw1.z*tempd1.z;
      //resCache1[jj*out_2+kk] += wmatPtr[ii*dkernel_01+mm*kernel_01+pp*kernel_1+tt]* \
              dataP[mm*d_12+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];

      //float3 tempw2 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+1*kernel_1]));
      float3 tempd2 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw2.x*tempd2.x;
      sum2 += tempw2.y*tempd2.y;
      sum2 += tempw2.z*tempd2.z;

      //float3 tempw3 = (*((__global float3*)&wmatPtr[ii*dkernel_01+mm*kernel_01+2*kernel_1]));
      float3 tempd3 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw3.x*tempd3.x;
      sum3 += tempw3.y*tempd3.y;
      sum3 += tempw3.z*tempd3.z;*/

      int kk=0;
      //float3 tempd1 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      float3 tempd1 = data1.s012;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      //float3 tempd2 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      float3 tempd2 = data2.s012;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      //float3 tempd3 = (*((__global float3*)&dataP[mm*d_12+(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      float3 tempd3 = data3.s012;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      //resCache1[jj*out_2+kk] += sum1+sum2+sum3;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=1;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s123;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s123;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s123;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=2;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s234;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s234;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s234;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=3;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s345;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s345;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s345;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=4;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s456;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s456;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s456;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=5;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s567;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s567;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s567;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=6;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s678;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s678;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s678;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=7;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s789;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s789;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s789;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=8;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s89a;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s89a;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s89a;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=9;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.s9ab;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.s9ab;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.s9ab;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=10;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.sabc;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.sabc;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.sabc;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;

      kk=11;sum1=0.0;sum2=0.0;sum3=0.0;
      tempd1 = data1.sbcd;//(*((__local float3*)&dataCache[(kernel_stride0*jj+0)*d_2+kernel_stride1*kk]));
      sum1 += tempw1.s0*tempd1.x;      sum1 += tempw1.s1*tempd1.y;      sum1 += tempw1.s2*tempd1.z;
      tempd2 = data2.sbcd;//(*((__local float3*)&dataCache[(kernel_stride0*jj+1)*d_2+kernel_stride1*kk]));
      sum2 += tempw1.s3*tempd2.x;      sum2 += tempw1.s4*tempd2.y;      sum2 += tempw1.s5*tempd2.z;
      tempd3 = data3.sbcd;//(*((__local float3*)&dataCache[(kernel_stride0*jj+2)*d_2+kernel_stride1*kk]));
      sum3 += tempw1.s6*tempd3.x;      sum3 += tempw1.s7*tempd3.y;      sum3 += tempwl*tempd3.z;
      resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] +=sum1+sum2+sum3;
  }
}

//barrier( CLK_LOCAL_MEM_FENCE );
/*for(int jj=0;jj<out_1;jj++){
//for(int kk=0;kk<out_2;kk+=4)
  //resPtr[i*ls*out_12+j*out_12+jj*out_2+kk] = resCache[j*out_1*out_2+jj*out_2+kk];
  (*((__global float4*)&resPtr[i*ls*out_12+j*out_12+jj*out_2])) = (*((__private float4*)&resCache1[jj*out_2]));
  (*((__global float8*)&resPtr[i*ls*out_12+j*out_12+jj*out_2+4])) = (*((__private float8*)&resCache1[jj*out_2+4]));
}*/

#endif


#ifdef Opti13
const int ii = get_group_id(0);
const int jj = get_local_id(0);

const int stride0 = 1,stride1=1;
const int pad0=1, pad1=1;

/*if(jj==1)
{
  for(int l1=0;l1<d_1;l1++)
    for(int l2=0;l2<d_2;l2++)
      dataCache[l1*d_2+l2] = 0.0;
}*/


for(int kk=0;kk<out_2;kk++) {
  float tempAcc = 0.0;
  //resCache[jj*out_2+kk] = 0.0;
  //resPtr[ii*out_1*out_2+jj*out_2+kk] = 0.0;
  for(int mm=0;mm<d_0;mm++)
  {
    if(jj==0) {
      for(int i1=0;i1<kernel_0;i1++)
        for(int j1=0;j1<kernel_1;j1++)
          //wmatCache[i1][j1] = wmatPtr[ii][mm][i1][j1];
          wmatCache[i1*kernel_1+j1] = wmatPtr[ii*d_0*kernel_0*kernel_1+mm*kernel_0*kernel_1+i1*kernel_1+j1];
    }
    //barrier( CLK_LOCAL_MEM_FENCE );
      //resCache[jj][kk] = 0.0;
      //dataCache[jj+2*pad0][kk+2*pad1] = dataP[mm][jj+2*pad0][kk+2*pad1];

      //dataCache[(jj+pad0)*d_2+kk+pad1] = dataP[mm*d_1*d_2+(jj+pad0)*d_2+kk+pad1];
      if(jj==1)
      {
        for(int l1=0;l1<d_1;l1++)
          for(int l2=0;l2<d_2;l2++)
            dataCache[l1*d_2+l2] = dataP[mm*d_1*d_2+l1*d_2+l2];
      }
      barrier( CLK_LOCAL_MEM_FENCE );
    for(int pp=0;pp<kernel_0;pp++)
      for(int tt=0;tt<kernel_1;tt++)
        //resCache[jj][kk] += wmatCache[pp][tt]*dataCache[jj*stride0+pp][kk*stride1+tt];
        //resCache[jj*out_2+kk] += wmatCache[pp*kernel_1+tt]*dataCache[(jj*stride0+pp)*d_2+kk*stride1+tt];
        tempAcc += wmatCache[pp*kernel_1+tt]*dataCache[(jj*stride0+pp)*d_2+kk*stride1+tt];
        //tempAcc += wmatCache[pp*kernel_1+tt]*dataP[mm*d_1*d_2+(kernel_stride0*jj+pp)*d_2+kernel_stride1*kk+tt];
    //barrier( CLK_LOCAL_MEM_FENCE );
    //resPtr[ii*out_1*out_2+jj*out_2+kk] = resCache[jj][kk];


  }
  resPtr[ii*out_1*out_2+jj*out_2+kk] = tempAcc;
  //resPtr[ii*out_1*out_2+jj*out_2+kk] = resCache[jj*out_2+kk];
}

#endif

}


/*    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    int resultIndex = gid*M;
    int iters = 0;
    int fillFlag = K%4;

    if (gid < N){
      for(int j=0;j<M;j++){
        //use local Memory to cache a_trans's entire col
        int offset = j*K;
        for (int k = lid;k < K;k += lsize ){
          //dataCacheA[k] = A[k+offset];
          *((__local float*)&dataCacheA[k]) = *((const __global float*)&A[k+offset]);
        }
        barrier( CLK_LOCAL_MEM_FENCE );


        int indexB = gid*K;
        int indexA = 0;
        float sum =0.0f;
        if (fillFlag==0){
          for (int h =0;h<K;h+=4){
            //sum +=dot(vload4(0,indexB),vload4(0,indexA));
            float4 tmpb = (*((__global float4*)&B[indexB]));
            float4 tmpa = (*((__local float4*)&dataCacheA[indexA]));
            sum += tmpa.x * tmpb.x;
            sum += tmpa.y * tmpb.y;
            sum += tmpa.z * tmpb.z;
            sum += tmpa.w * tmpb.w;
            //sum += dot(tmpa,tmpb);
            indexA+=4;
            indexB+=4;
          }
        }
        else{
          for(int h=0;h<K-fillFlag;h+=4){
            float4 tmpb = (*((__global float4*)&B[indexB]));
            float4 tmpa = (*((__local float4*)&dataCacheA[indexA]));
            //sum += dot(tmpa,tmpb);
            sum += tmpa.x * tmpb.x;
            sum += tmpa.y * tmpb.y;
            sum += tmpa.z * tmpb.z;
            sum += tmpa.w * tmpb.w;
            indexA+=4;
            indexB+=4;
          }
          for (int r=0;r<fillFlag;r++){
            sum += B[indexB+r]*dataCacheA[indexA+r];
          }
        }
        C[resultIndex+j]=sum;
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
}*/
