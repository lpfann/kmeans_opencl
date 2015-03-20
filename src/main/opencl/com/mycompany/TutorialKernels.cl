
__kernel void find_nearest_prototype(
__global const float* data,
__global const float* prototypes,
__global uint* out_assignment,
const uint dim,
const uint K,
uint n)
{

    const int i = get_global_id(0);
    if (i >= n)
        return;

    float min = FLT_MAX;
    
    uint nearestproto = 0;
    float sum;
    for (uint k = 0; k < K; k++){

        sum = 0.0f;
        for (uint d = 0; d < dim;d++){
            sum += pown( data[i*dim +d] - prototypes[k*dim+d], 2);
        }
        //sum = native_sqrt(sum);
        min = fmin(min,sum);
        nearestproto = (min==sum) ? k : nearestproto;

    }
    out_assignment[i] = nearestproto;
}


__kernel void calc_prototype(
__global const float* data,
__global const uint* assignment,
__global float* out_prototypes,
__global uint* out_count,
const uint dim,
const uint K,
const uint N)
{

    const int i = get_global_id(0);
    if (i >= K*dim)
        return;
    const uint k = i/dim;
    const uint d = i%dim;

    float element = 0.0f;
    uint count = 0;
    for (uint x = 0; x < N; x++){

        if(assignment[x]==k){
            element += data[x*dim +d];
            count++;

        }

    }
    if(count==0){
        return;
    }


    barrier(CLK_GLOBAL_MEM_FENCE);
    if(d==0){
        out_count[k] = count;
    }

    element = native_divide(element,out_count[k]);
    out_prototypes[i] = element;
    //printf("d:%d k:%d count:%d element:%f\n",d,k,count,element);
}
