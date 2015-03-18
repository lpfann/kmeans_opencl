package com.mycompany;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.util.OpenCLType;
import com.nativelibs4java.opencl.util.ReductionUtils;
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
import java.util.stream.Collector;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.groupingByConcurrent;

public class main {
    private static final int MAX_ITERATIONS = 1000;
    static float[] data;
    static ArrayList<String> ids;
    public static final int K = 2;
    public static int DIM;
    public static int N;

    public static void main(String[] args) throws IOException {
//        String path = "C:\\Users\\mirek_000\\Documents\\Dropbox\\workspace\\Clustering_Project\\GEOdata_Cholesteatom_nurLogRatio.csv";
        String path = "C:\\Users\\mirek_000\\Documents\\Dropbox\\workspace\\Clustering_Project\\Test.csv";
        importCSV(path);

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

        int[] nearestPrototypeforVector = new int[N];
        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();

        CLBuffer<Float> vectors = context.createFloatBuffer(Usage.Input, FloatBuffer.wrap(data), true);
        CLBuffer<Float> prototypeBuffer = context.createFloatBuffer(Usage.Input, FloatBuffer.wrap(prototypes), false);
        CLBuffer<Integer> proto_Assignment = context.createIntBuffer(Usage.Output, IntBuffer.wrap(nearestPrototypeforVector), false);

//        // Read the program sources and compile them :
//        String src = IOUtils.readText(com.mycompany.main.class.getResource("TutorialKernels.cl"));
//        CLProgram program = context.createProgram(src);
//        // Get and call the kernel :
//        CLKernel addFloatsKernel = program.createKernel("find_nearest_prototype");
        long t0 = System.currentTimeMillis();

        TutorialKernels kernels = new TutorialKernels(context);
        int[] globalSizes = new int[] {N};
        float[] newprototypes;
        CLEvent findNearestPrototypes;
        Pointer<Integer> outPtr;
        int[] selectedPrototypes;
        for (int t = 0; t < MAX_ITERATIONS; t++) {
            findNearestPrototypes = kernels.find_nearest_prototype(queue, vectors, prototypeBuffer, proto_Assignment, DIM, K, N, globalSizes, null);

            outPtr = proto_Assignment.read(queue, findNearestPrototypes);
            selectedPrototypes = outPtr.getInts();
            newprototypes = new float[DIM * K];
            for (int i = 0; i < N; i++) {
                for (int d = 0; d < DIM; d++) {
                    newprototypes[selectedPrototypes[i] * DIM + d] += data[i + d];
                }
            }
            for (int k = 0; k < K; k++) {
                final int filter = k;
                long count= Arrays.stream(selectedPrototypes).parallel().filter((s->s == filter)).count();
                System.out.print("count:"+count + " ");
                for (int d = 0; d < DIM; d++) {
                    newprototypes[k* DIM +d] /= count;
                }
            }

            prototypes = newprototypes;
            prototypeBuffer.release();
            prototypeBuffer = context.createFloatBuffer(Usage.Input, FloatBuffer.wrap(prototypes), false);
            proto_Assignment.release();
            proto_Assignment = context.createIntBuffer(Usage.Output, IntBuffer.wrap(nearestPrototypeforVector), false);

            System.out.print(selectedPrototypes[0] + " ");
            System.out.println(prototypes[selectedPrototypes[0]]+"x "+ prototypes[selectedPrototypes[0]+1]+"y");
        }
        long t1 = System.currentTimeMillis();
        System.out.println(t1 - t0 + "ms");

    }


    private static void importCSV(String path) {

        try {
            Reader in = new FileReader(path);
            List<CSVRecord> records = CSVFormat.DEFAULT.parse(in).getRecords();  // Tab Delimiter

            DIM = records.get(0).size()-1;   // -1 ID column
            N = records.size()-1;     // -1 Header Row
            ids = new ArrayList<>(N);
            data = new float[DIM * N];
            NumberFormat nf = NumberFormat.getInstance(Locale.GERMAN);

            for (int j = 1; j < N; j++) {
                CSVRecord record = records.get(j);
                ids.add(record.get(0));

                for (int i = 1; i < DIM+1; i++) {
                    data[(j-1)* DIM + i-1]= nf.parse(record.get(i)).floatValue();
                }

            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        } finally {

        }
    }
}
