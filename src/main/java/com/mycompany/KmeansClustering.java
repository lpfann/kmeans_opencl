package com.mycompany;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.util.OpenCLType;
import com.nativelibs4java.opencl.util.ReductionUtils;
import org.bridj.Pointer;

import javax.sql.rowset.serial.SerialRef;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;


public class KmeansClustering {
    private static final int MAX_ITERATIONS = 100000000;
    static int K;
    static int N;
    private final int k;
    private final int PROTOTYPE_SIZE;
    private final int DATA_SIZE;
    CLContext context;
    CLQueue queue;
    Pointer<Float> dataPointsPtr;
    CLBuffer<Float> dataPointsBuffer; // True means, data can be cached
    Pointer<Float> prototypePtr;
    CLBuffer<Float> prototypeBuffer;
    Pointer<Integer> assignmentPtr;
    CLBuffer<Integer> proto_Assignment;
    Pointer<Integer> outPtr;
    CLEvent findNearestPrototypesEvent;
    CLEvent calcPrototypesEvent;
    CLEvent readAssignmentsEvent;
    CLEvent writenewdata;
    private float[] data;
    private int dim;
    private int n;
    private float[] prototypes;
    private int[] clusterForEachPoint;
    private int[] new_clusterForEachPoint;
    private TutorialKernels kernels;


    public KmeansClustering(int k, int dim, float[] data, boolean useCPU) {
        this.data = data;
        this.k = k;
        this.dim = dim;
        this.n = data.length / dim;
        this.DATA_SIZE = n * dim;
        this.PROTOTYPE_SIZE = k * dim;

        // OpenCL Init.
        context = JavaCL.createBestContext(useCPU ? CLPlatform.DeviceFeature.CPU : CLPlatform.DeviceFeature.GPU);
        System.err.println("Used Device: "+context.getDevices()[0].getName());
        queue = context.createDefaultQueue();

        // Load Kernel
        kernels = null;
        try {
            kernels = new TutorialKernels(context);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }

        // Init. Buffers for data Points
        dataPointsPtr = Pointer.allocateFloats(DATA_SIZE).order(context.getByteOrder());
        dataPointsPtr.setFloats(data);
        dataPointsBuffer = context.createBuffer(Usage.Input, dataPointsPtr, true);

        // Init Prototype Buffers  and location of first generation
        prototypePtr = Pointer.allocateFloats(PROTOTYPE_SIZE);
        //prototypes = initPrototypes();
        prototypes = initPrototypesKMeansPP();
        prototypePtr.setFloats(prototypes);
        prototypeBuffer = context.createBuffer(Usage.InputOutput, prototypePtr, false);

        // Init Buffer for Cluster assignments
        assignmentPtr = Pointer.allocateInts(n);
        clusterForEachPoint = new int[n];
        new_clusterForEachPoint = new int[n];
        proto_Assignment = context.createBuffer(Usage.InputOutput, assignmentPtr, false);

        // Init device memory for output
        outPtr = Pointer.allocateInts(n);




    }

    public int[] clusteringLoop() {

        boolean finished = false;

        /**
         * Main KMeans Loop - Runs until convergence to steady cluster assignments for each point
         */
        int t = 0;
        int old_change_val = 0;
        final int change_threshold = 100;
        int change_counter = 0;
        while (++t < MAX_ITERATIONS) {

            // Main Call for Computing Kernel - Runs Distance Meaasure for each point to each Cluster Prototype
            findNearestPrototypesEvent = kernels.find_nearest_prototype(queue, dataPointsBuffer, prototypeBuffer, proto_Assignment, dim, k, n, new int[]{n}, null);   // Expectation Step ( EM-Algorithm)

            // Read results when previous call finished
            readAssignmentsEvent = proto_Assignment.read(queue, outPtr, true, findNearestPrototypesEvent);
            new_clusterForEachPoint = outPtr.getInts();

            //float[] read = prototypeBuffer.read(queue, findNearestPrototypesEvent).getFloats();

            // Convergence if no assignments changed
            if (Arrays.equals(new_clusterForEachPoint, clusterForEachPoint)) {
                finished = true;
                break;
            }


            //Check Convergence - if nothing changes for a specified amount of iterations - break out
            int count=0;
            for (int i = 0; i < this.n; i++) {
                if (clusterForEachPoint[i]!= new_clusterForEachPoint[i]) {
                    count++;
                }
            }
            int delta = Math.abs(count - old_change_val);
            if (delta == 0) {
                if (change_counter > change_threshold) {
                    break;
                }
                change_counter++;
            }
            old_change_val = count;

            // Update Assignments
            clusterForEachPoint = new_clusterForEachPoint;

            //Calculate new Prototype positions
            //calcPrototypesEvent = kernels.calc_prototype(queue, dataPointsBuffer, proto_Assignment, prototypeBuffer, dim, k, n, new int[]{k * dim}, null);
            //calcPrototypesEvent.waitFor();
            //prototypes = prototypeBuffer.read(queue,calcPrototypesEvent).getFloats(k*dim);

            prototypes = calcNewPrototypes(clusterForEachPoint, k, n, dim, data, prototypes);

            try {
                Pointer<Float> data = prototypeBuffer.map(queue, CLMem.MapFlags.Write);
                data.setFloats(prototypes);
                prototypeBuffer.unmap(queue, data);
            } catch (CLException.MapFailure ex) {
                Pointer<Float> newprotpointer = Pointer.allocateFloats(PROTOTYPE_SIZE);
                newprotpointer.setFloats(prototypes);
                prototypeBuffer.write(queue, newprotpointer, true);
            }
        }

        if (!finished) {
            System.err.println("Max-Iterations exceeded.");
        }
        return new_clusterForEachPoint;
    }

    /**
     * Calculates new positition for all 'K' Cluster Prototypes
     * @return New Positions in 1-DIM Float Array (Each Point has DIM Elements)
     * @param clusterForEachPoint
     * @param k
     * @param n
     * @param dim
     * @param data
     * @param old_prototypes
     */
    public float[] calcNewPrototypes(int[] clusterForEachPoint, int k, int n, int dim, float[] data, float[] old_prototypes) {
        float[] newprototypes;
        int prototype_size = k * dim;
        newprototypes = new float[prototype_size];
        long[] counts = new long[k];
        // Sum up all Points in each Cluster
        for (int i = 0; i < n; i++) {
            // Count all points for each Cluster
            counts[clusterForEachPoint[i]]++;
            // Sum
            for (int j = 0; j < dim; j++) {
                // Elementwise summation
                newprototypes[clusterForEachPoint[i] * dim + j] += data[i + j];
            }
        }

        // Dividie Sum by Cluster Element Count for the Mean
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                newprototypes[i * dim + j] = (newprototypes[i * dim + j] + old_prototypes[i * dim + j]) / (counts[i] + 1);
            }
        }
        return newprototypes;
    }

    /**
     * Initialising prototype positions
     * Randomly choosing unique Indices for now
     *
     * @return Prototype Positions
     */
    public float[] initPrototypes() {
        // Select Random Prototypes from Data
        Random random = new Random();
        float[] prototypes = new float[PROTOTYPE_SIZE];
        int[] prototypeIDs = new int[k];
        for (int i = 0; i < k; i++) {
            // Select Random Unique Indices for all K
            boolean randomNumberUnique;
            do {
                int index = random.nextInt(n);
                randomNumberUnique = true;
                prototypeIDs[i] = index;
                for (int j = 0; j < i; j++) {
                    if (prototypeIDs[j] == index) {
                        randomNumberUnique = false;
                        break;
                    }
                }
            } while (!randomNumberUnique);
            // Copy Raw Values into Float Array
            System.arraycopy(data, prototypeIDs[i] * dim, prototypes, i * dim, dim);
        }
        ///
        return prototypes;
    }

    /**
     * Initialising prototype positions  using Kmeans++
     *
     * @return Prototype Positions
     */
    public float[] initPrototypesKMeansPP() {
        // Select Random Prototypes from Data
        Random random = new Random();
        float[] prototypes = new float[PROTOTYPE_SIZE];
        prototypePtr = Pointer.allocateFloats(PROTOTYPE_SIZE);
        prototypePtr.setFloats(prototypes);
        prototypeBuffer = context.createBuffer(Usage.InputOutput, prototypePtr, false);

        // Init Dist buffer
        Pointer<Float> distPointer = Pointer.allocateFloats(n);
        CLBuffer<Float> distBuffer = context.createBuffer(Usage.InputOutput, distPointer, false);
        ReductionUtils.Reductor<Float> reductor = ReductionUtils.createReductor(context, ReductionUtils.Operation.Add, OpenCLType.Float, 1);

        int[] prototypeIDs = new int[k];
        float sum = 0;
        for (int i = 0; i < k; i++) {
            if (i == 0) {
                // Select First prototype
                int index = random.nextInt(n);
                prototypeIDs[i] = index;
            } else {

                CLEvent calc_dist = kernels.calc_dist_to_nearest_prototype(queue, dataPointsBuffer, prototypeBuffer, distBuffer, dim, k, n, i, new int[]{n}, null);
                CLEvent read = distBuffer.read(queue, distPointer, true, calc_dist);
                float[] dists = distPointer.getFloats();
                Pointer<Float> out = Pointer.allocateFloat();


                CLEvent reduce = reductor.reduce(queue, distBuffer, n, out, n);
                reduce.waitFor();
                sum = out.getFloat();

                Arrays.sort(dists);
                float chosen_bound = random.nextFloat()*sum;
                float accumulator = 0.0f;
                int pos = 0;
                while (accumulator < chosen_bound) {
                    accumulator+=dists[pos];
                    pos++;

                }
                if (pos == n) {
                    pos -=1;
                }
                prototypeIDs[i]=pos;
                assert pos < n;
            }

                // Copy Raw Values into Float Array

                System.arraycopy(data, prototypeIDs[i] * dim, prototypes, i * dim, dim);

            try {
                Pointer<Float> data = prototypeBuffer.map(queue, CLMem.MapFlags.Write);
                data.setFloats(prototypes);
                prototypeBuffer.unmap(queue, data);
            } catch (CLException.MapFailure ex) {
                Pointer<Float> newprotpointer = Pointer.allocateFloats(PROTOTYPE_SIZE);
                newprotpointer.setFloats(prototypes);
                prototypeBuffer.write(queue, newprotpointer, true);
            }
        }
        ///
        System.err.println("Initalized prototypes using Kmeans++ behaviour");
        return prototypes;
    }


}

