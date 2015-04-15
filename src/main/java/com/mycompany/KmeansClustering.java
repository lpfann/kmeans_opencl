package com.mycompany;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import org.apache.commons.cli.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.bridj.Pointer;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.FloatBuffer;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.*;


public class KmeansClustering {
    private static final int MAX_ITERATIONS = 1000;
    private static int K;
    private static float[] data;
    private static ArrayList<String> ids;
    private static int DIM;
    private static int N;

    public static void main(String[] args) throws IOException {
        String path = "";
        Options options = new Options();
        options.addOption(OptionBuilder.withLongOpt("numberOfClusters").withDescription("Number of clusters").withType(Number.class).hasArg().withArgName("k").create());
        options.addOption(OptionBuilder.withLongOpt("path").withDescription("Path to Input File").withType(String.class).hasArg().withArgName("p").create());
        CommandLineParser commandLineParser = new PosixParser();
        try {
            CommandLine commandLine = commandLineParser.parse(options, args);
            int value = 0;
            if (commandLine.hasOption("numberOfClusters")) {
                value = ((Number) commandLine.getParsedOptionValue("numberOfClusters")).intValue();
                if (value > 1) {
                    K = value;
                } else {
                    System.out.println("K Value smaller than 2. Choose a better value");
                    System.exit(-1);
                }
            } else {
                System.out.println("K Value is missing.");
                System.exit(-1);
            }
            if (commandLine.hasOption("path")) {
                path = (String) commandLine.getParsedOptionValue("path");
                System.out.println();

            } else {
                System.out.println("Missing Path to .csv file.");
                System.exit(-1);
            }

        } catch (org.apache.commons.cli.ParseException e) {
            e.printStackTrace();
        }

        importCSV(path);

        //DEBUG CODE
//        // Read the program sources and compile them :
//        String src = IOUtils.readText(com.mycompany.main.class.getResource("TutorialKernels.cl"));
//        CLProgram program = context.createProgram(src);
//        // Get and call the kernel :
//        CLKernel addFloatsKernel = program.createKernel("find_nearest_prototype");

        //       long t0 = System.currentTimeMillis();
        startClustering(K, DIM, N);
        //       long t1 = System.currentTimeMillis();
        //       System.out.println(t1 - t0 + "ms");
    }

    private static int[] startClustering(int k, int dim, int n) {
        CLContext context = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU);
        CLQueue queue = context.createDefaultQueue();
        // Init. Buffers for first call
        CLBuffer<Float> vectors = context.createFloatBuffer(Usage.Input, FloatBuffer.wrap(data), true);
        Pointer<Float> prototypePtr = Pointer.allocateFloats(k * dim);
        float[] initialProtos = initPrototypes();
        prototypePtr.setFloats(initialProtos);
        CLBuffer<Float> prototypeBuffer = context.createFloatBuffer(Usage.InputOutput, prototypePtr, false);
        Pointer<Integer> clusters = Pointer.allocateInts(n);
        CLBuffer<Integer> proto_Assignment = context.createIntBuffer(Usage.InputOutput, clusters, false);
        // Load Kernel
        TutorialKernels kernels = null;
        try {
            kernels = new TutorialKernels(context);
        } catch (IOException e) {
            e.printStackTrace();
        }
        CLEvent findNearestPrototypes;
        CLEvent calcPrototypes;
        CLEvent readData;
        CLBuffer<Integer> countBuffer = context.createIntBuffer(Usage.Input, k);
        Pointer<Integer> outPtr = Pointer.allocateInts(n);
        int[] clusterForEachPoint = new int[n];
        int[] new_clusterForEachPoint;
        CLEvent writenewdata = prototypeBuffer.write(queue, prototypePtr, true);
        boolean finished = false;

        /**
         * Main KMeans Loop - Runs until convergence to steady cluster assignments for each point
         */
        int t = 0;
        while (++t < MAX_ITERATIONS) {
            // Main Call for Computing Kernel - Runs Distance Meaasure for each point to each Cluster Prototype
            findNearestPrototypes = kernels.find_nearest_prototype(queue, vectors, prototypeBuffer, proto_Assignment, dim, k, n, new int[]{n}, null, writenewdata);   // Expectation Step ( EM-Algorithm)
            // Read results when previous call finished
            readData = proto_Assignment.read(queue, outPtr, true, findNearestPrototypes);
            new_clusterForEachPoint = outPtr.getInts();
            float[] read = prototypeBuffer.read(queue, findNearestPrototypes).getFloats();
            // Convergence if no assignments changed
            if (Arrays.equals(new_clusterForEachPoint, clusterForEachPoint)) {
                clusterForEachPoint = new_clusterForEachPoint;
                finished = true;
                break;
            }
            clusterForEachPoint = new_clusterForEachPoint;

            //Calculate new Prototype positions
            calcPrototypes = kernels.calc_prototype(queue, vectors, proto_Assignment, prototypeBuffer, countBuffer, dim, k, n, new int[]{k * dim}, null);

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
        return clusterForEachPoint;
    }

    /**
     * Initalising prototype positions
     * Randomly choosing unique Indices for now
     * TODO implement init. from Kmeans++
     *
     * @return Prototype Positions
     */
    private static float[] initPrototypes() {
        // Select Random Prototypes from Data
        Random random = new Random();
        float[] prototypes = new float[DIM * K];
        int[] prototypeIDs = new int[K];
        for (int k = 0; k < K; k++) {
            // Select Random Unique Indices for all K
            boolean randomNumberUnique;
            do {
                int index = random.nextInt(N);
                randomNumberUnique = true;
                prototypeIDs[k] = index;
                for (int j = 0; j < k; j++) {
                    if (prototypeIDs[j] == index) {
                        randomNumberUnique = false;
                        break;
                    }
                }
            } while (!randomNumberUnique);
            // Copy Raw Values into Float Array
            System.arraycopy(data, prototypeIDs[k] * DIM, prototypes, k * DIM, DIM);
        }
        ///
        return prototypes;
    }

    /**
     * Import CSV Files and store it in data array
     *
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

            for (int j = 1; j < N + 1; j++) {
                CSVRecord record = records.get(j);
                ids.add(record.get(0));

                for (int i = 1; i < DIM + 1; i++) {
                    data[(j - 1) * DIM + i - 1] = nf.parse(record.get(i)).floatValue();
                }

            }
        } catch (IOException | ParseException e) {
            e.printStackTrace();
            System.out.println("Loading of InputFile was not possible. Exiting...");
            System.exit(-1);
        }
    }
}

