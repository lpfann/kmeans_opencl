package com.mycompany;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.bridj.Pointer;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.*;

import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.groupingByConcurrent;

public class main {
    private static final int MAX_ITERATIONS = 300;
    static float[] data;
    static ArrayList<String> ids;
    public static final int K = 100;
    public static int DIM;
    public static int N;

    public static void main(String[] args) throws IOException {
        String path = "C:\\Users\\mirek_000\\Documents\\Dropbox\\workspace\\Clustering_Project\\GEOdata_Cholesteatom_nurLogRatio.csv";
//        String path = "C:\\Users\\mirek_000\\Documents\\Dropbox\\workspace\\Clustering_Project\\Test.csv";
//        String path = "/home/lukas/workspace/kmeans_clustering_opencl/GEOdata_Cholesteatom_nurLogRatio.csv";
//        String path = "/home/lukas/workspace/kmeans_clustering_opencl/Test.csv";
        importCSV(path);

          //DEBUG CODE
//        // Read the program sources and compile them :
//        String src = IOUtils.readText(com.mycompany.main.class.getResource("TutorialKernels.cl"));
//        CLProgram program = context.createProgram(src);
//        // Get and call the kernel :
//        CLKernel addFloatsKernel = program.createKernel("find_nearest_prototype");

        long t0 = System.currentTimeMillis();

        CLContext context = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU);
        CLQueue queue = context.createDefaultQueue();
        // Init. Buffers for first call
        CLBuffer<Float> vectors = context.createFloatBuffer(Usage.Input, FloatBuffer.wrap(data), true);
        Pointer<Float> prototypePtr = Pointer.allocateFloats(K * DIM);
        prototypePtr.setFloats(initPrototypes());
        CLBuffer<Float> prototypeBuffer = context.createFloatBuffer(Usage.InputOutput, prototypePtr, false);
        Pointer<Integer> clusters = Pointer.allocateInts(N);
        CLBuffer<Integer> proto_Assignment = context.createIntBuffer(Usage.InputOutput, clusters, false);;
        // Load Kernel
        TutorialKernels kernels = new TutorialKernels(context);
        float[] newPrototypes;
        CLEvent findNearestPrototypes;
        CLEvent calcPrototypes;
        CLEvent readData;
        CLBuffer<Integer> countBuffer = context.createIntBuffer(Usage.Input, K);
        Pointer<Integer> outPtr = Pointer.allocateInts(N);
        int[] clusterForEachPoint = new int[DIM * N];
        int[] new_clusterForEachPoint;
        CLEvent writenewdata = prototypeBuffer.write(queue, prototypePtr, true);
        boolean finished = false;
        int t = 0;
        /**
         * Main KMeans Loop - Runs until convergence to steady cluster assignments for each point
         */
        while (!finished && ++t < MAX_ITERATIONS ) {
            // Main Call for Computing Kernel - Runs Distance Meaasure for each point to each Cluster Prototype
            findNearestPrototypes = kernels.find_nearest_prototype(queue, vectors, prototypeBuffer, proto_Assignment, DIM, K, N, new int[]{N}, null, writenewdata);   // Expectation Step ( EM-Algorithm)
            // Read results when previous call finished
            readData = proto_Assignment.read(queue, outPtr, true, findNearestPrototypes);
            new_clusterForEachPoint = outPtr.getInts();

            // Convergence if no assignments changed
            if (Arrays.equals(new_clusterForEachPoint, clusterForEachPoint)) {
                finished=true;
                break;
            }
            clusterForEachPoint = new_clusterForEachPoint;

            //Calculate new Prototype positions

            calcPrototypes = kernels.calc_prototype(queue, vectors, proto_Assignment, prototypeBuffer, countBuffer, DIM, K, N, new int[]{K * DIM}, null);
//            readData = prototypeBuffer.read(queue,prototypePtr,true,calcPrototypes);
//            newPrototypes = prototypePtr.getFloats();

            // Write back to Device Memory
//            prototypePtr.setFloats(newPrototypes);
//            writenewdata = prototypeBuffer.write(queue,prototypePtr,true);

           // DEBUG CODE
//            System.out.print(new_clusterForEachPoint[100] + " ");
//            System.out.println(newPrototypes[new_clusterForEachPoint[100]]+"x "+ newPrototypes[new_clusterForEachPoint[100]+1]+"y");
        }

        if (!finished) {
            System.out.println("Max-Iterations exceeded. Results did not converge.");
        }
        long t1 = System.currentTimeMillis();
        System.out.println(t1 - t0 + "ms");

    }

    /**
     * Calculates new positition for all 'K' Cluster Prototypes
     * @param clusterForEachPoint Assignment for each point in Dataset to cluster
     * @return New Positions in 1-DIM Float Array (Each Point has DIM Elements)
     */
    private static float[] calcNewPrototypes(int[] clusterForEachPoint) {
        float[] newprototypes;
        newprototypes = new float[DIM * K];
        long[] counts = new long[K];
        // Sum up all Points in each Cluster


        for (int i = 0; i < N; i++) {
            // Count all points for each Cluster
            counts[clusterForEachPoint[i]]++;
            // Sum
            for (int d = 0; d < DIM; d++) {
                // Elementwise summation
                newprototypes[clusterForEachPoint[i] * DIM + d] += data[i + d];
            }
        }

        // Dividie Sum by Cluster Element Count for the Mean
        for (int k = 0; k < K; k++) {
            for (int d = 0; d < DIM; d++) {
                newprototypes[k* DIM +d] /= counts[k];
            }
        }
        return newprototypes;
    }

    /**
     * Initalising prototype positions
     * Randomly choosing unique Indices for now
     * TODO implement init. from Kmeans++
     * @return Prototype Positions
     */
    private static float[] initPrototypes() {
        // Select Random Prototypes from Data
        Random random = new Random();
        float[] prototypes = new float[DIM *K];
        int[] prototypeIDs = new int[K];
        for (int k = 0; k < K; k++) {
            // Select Random Unique Indices for all K
            boolean randomNumberUnique = true;
                do {
                    int index = random.nextInt(N);
                    prototypeIDs[k] = index;
                    for (int j = 0; j < k; j++) {
                        if (prototypeIDs[j] == index) {
                            randomNumberUnique = false;
                        }
                    }
                } while (!randomNumberUnique);
            // Copy Raw Values into Float Array
            for (int d = 0; d < DIM; d++) {
                prototypes[k* DIM +d] = data[prototypeIDs[k]+d];
            }
        }
        ///
        return prototypes;
    }

    /**
     * Import CSV Files and store it in data array
     * @param path
     */
    private static void importCSV(String path) {

        try {
            Reader in = new FileReader(path);
            // CSVFormat.TDF for Tabs or CSVFormat.DEFAULT for Commas
            List<CSVRecord> records = CSVFormat.TDF.parse(in).getRecords();  // Tab Delimiter

            DIM = records.get(0).size() - 1;   // -1 ID column
            N = records.size() - 1;     // -1 Header Row
            ids = new ArrayList<>(N); // Store Names for each Point
            data = new float[DIM * N];
            NumberFormat nf = NumberFormat.getInstance(Locale.GERMAN);

            for (int j = 1; j < N; j++) {
                CSVRecord record = records.get(j);
                ids.add(record.get(0));

                for (int i = 1; i < DIM + 1; i++) {
                    data[(j - 1) * DIM + i - 1] = nf.parse(record.get(i)).floatValue();
                }

            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("Loading of InputFile was not possible. Exiting...");
            System.exit(-1);
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Loading of InputFile was not possible. Exiting...");
            System.exit(-1);
        } catch (ParseException e) {
            e.printStackTrace();
            System.out.println("Loading of InputFile was not possible. Exiting...");
            System.exit(-1);
        }
    }
}

