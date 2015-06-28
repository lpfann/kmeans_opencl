package com.mycompany;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import org.apache.commons.cli.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.bridj.Pointer;

import java.io.*;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.*;


public class KmeansClustering {
    private static final int MAX_ITERATIONS = 1000;
    public static boolean sout = false;
    static int K;
    static int DIM;
    static int N;
    private static ArrayList<String> ids;
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


    public KmeansClustering(int k, boolean useCPU, String path) {
        this.data = importCSV(path);
        this.k = k;

        this.DATA_SIZE = n * dim;
        this.PROTOTYPE_SIZE = k * dim;

        // OpenCL Init.
        context = JavaCL.createBestContext(useCPU ? CLPlatform.DeviceFeature.CPU : CLPlatform.DeviceFeature.GPU);
        queue = context.createDefaultQueue();

        // Init. Buffers for data Points

        dataPointsPtr = Pointer.allocateFloats(DATA_SIZE).order(context.getByteOrder());
        dataPointsBuffer = context.createBuffer(Usage.Input, dataPointsPtr, true);

        // Init Prototype Buffers  and location of first generation
        prototypePtr = Pointer.allocateFloats(PROTOTYPE_SIZE);
        prototypes = initPrototypes();
        prototypePtr.setFloats(prototypes);
        prototypeBuffer = context.createBuffer(Usage.InputOutput, prototypePtr, false);

        // Init Buffer for Cluster assignments
        assignmentPtr = Pointer.allocateInts(n);
        clusterForEachPoint = new int[n];
        new_clusterForEachPoint = new int[n];
        proto_Assignment = context.createBuffer(Usage.InputOutput, assignmentPtr, false);

        // Init device memory for output
        outPtr = Pointer.allocateInts(n);

        // Load Kernel
        kernels = null;
        try {
            kernels = new TutorialKernels(context);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }


    }

    public static void main(String[] args) throws IOException {
        String path = parseArgs(args);


        //       long t0 = System.currentTimeMillis();
        KmeansClustering kmeans = new KmeansClustering(K, false, path);
        int[] clustering = kmeans.clusteringLoop();
        if (sout) {
            for (int i = 0; i < N; i++) {
                System.out.println(clustering[i]);
            }
        } else {
            writeFile(clustering);
        }
        //       long t1 = System.currentTimeMillis();
        //       System.out.println(t1 - t0 + "ms");
    }

    private static String parseArgs(String[] args) {
        String path = "";
        Options options = new Options();
        options.addOption(OptionBuilder.withLongOpt("numberOfClusters").withDescription("Number of clusters").withType(Number.class).hasArg().withArgName("k").create());
        options.addOption(OptionBuilder.withLongOpt("path").withDescription("Path to Input File").withType(String.class).hasArg().withArgName("p").create());
        options.addOption("sout", false, "Print to system out");
        CommandLineParser commandLineParser = new PosixParser();
        HelpFormatter formatter = new HelpFormatter();

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
                formatter.printHelp("kmeans", options);
                System.exit(-1);
            }
            if (commandLine.hasOption("path")) {
                path = (String) commandLine.getParsedOptionValue("path");
                System.out.println();

            } else {
                System.out.println("Missing Path to .csv file.");
                formatter.printHelp("kmeans", options);
                System.exit(-1);
            }
            sout = commandLine.hasOption("sout");

        } catch (org.apache.commons.cli.ParseException e) {
            formatter.printHelp("kmeans", options);
            e.printStackTrace();
            System.exit(-1);
        }
        return path;
    }

    /**
     * Write Cluster Assignments to file.
     * Each line corresponds to one DataPoint
     *
     * @param clusterForEachPoint Data Array to be written
     * @throws IOException
     */
    private static void writeFile(int[] clusterForEachPoint) throws IOException {
        String filename = "out.dat";
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < N; i++) {
            bufferedWriter.write(clusterForEachPoint[i] + "");
            bufferedWriter.newLine();
        }
        bufferedWriter.flush();
        bufferedWriter.close();
    }

    private int[] clusteringLoop() {

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
                clusterForEachPoint = new_clusterForEachPoint;
                finished = true;
                break;
            }

/*
            //Check Convergence - if nothing changes for a specified amount of iterations - break out
            int count=0;
            LinkedList<Integer> outlier = new LinkedList<>();
            for (int i = 0; i < N; i++) {
                if (clusterForEachPoint[i]!= new_clusterForEachPoint[i]) {
                    count++;
                    outlier.add(i);
                }
            }

            // Update Assignments
            clusterForEachPoint = new_clusterForEachPoint;

            int delta = Math.abs(count - old_change_val);
            if (delta == 0) {
                if (change_counter > change_threshold) {
                    break;
                }
                change_counter++;
            }
            old_change_val = count;
            System.out.println(delta);
*/


            //Calculate new Prototype positions
            //calcPrototypesEvent = kernels.calc_prototype(queue, vectors, proto_Assignment, prototypeBuffer, countBuffer, dim, k, n, new int[]{k * dim}, null,findNearestPrototypesEvent);
            //calcPrototypesEvent.waitFor();

            prototypes = calcNewPrototypes();
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
            System.out.println("Max-Iterations exceeded. Results did not converge.");
        }
        return clusterForEachPoint;
    }

    /**
     * Calculates new positition for all 'K' Cluster Prototypes
     * @return New Positions in 1-DIM Float Array (Each Point has DIM Elements)
     */
    private float[] calcNewPrototypes() {
        float[] newprototypes;
        newprototypes = new float[PROTOTYPE_SIZE];
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
                newprototypes[i * dim + j] = (newprototypes[i * dim + j] + prototypes[i * dim + j]) / (counts[i] + 1);
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
    private float[] initPrototypes() {
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
     * Import CSV Files and store it in data array
     *
     * @param path
     */
    private float[] importCSV(String path) {

        try {
            Reader in = new FileReader(path);
            // CSVFormat.TDF for Tabs or CSVFormat.DEFAULT for Commas
            List<CSVRecord> records = CSVFormat.TDF.parse(in).getRecords();  // Tab Delimiter

            dim = records.get(0).size() - 1;   // -1 ID column
            DIM = dim;
            n = records.size() - 1;     // -1 Header Row
            N = n;
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
            return data;
        } catch (IOException | ParseException e) {
            e.printStackTrace();
            System.out.println("Loading of InputFile was not possible. Exiting...");
            System.exit(-1);

        }
        return null;
    }
}

