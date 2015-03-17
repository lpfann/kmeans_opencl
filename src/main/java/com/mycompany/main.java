package com.mycompany;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.library.OpenCLLibrary;
import com.nativelibs4java.opencl.util.*;
import com.nativelibs4java.util.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.bridj.Pointer;
import static org.bridj.Pointer.*;
import static java.lang.Math.*;
import org.apache.commons.lang.ArrayUtils;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Array;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.*;

public class main {
    private static final int MAX_ITERATIONS = 20;
    static float[] data;
    static ArrayList<String> ids;
    public static final int K = 10;
    public static int dim;
    public static int n;
    public static int[] nearestPrototypeforVector;

    public static void main(String[] args) throws IOException {
        String path = "C:\\Users\\mirek_000\\Documents\\Dropbox\\workspace\\Clustering_Project\\GEOdata_Cholesteatom_nurLogRatio.csv";
        importCSV(path);

        Random random = new Random();
        float[] prototypes = new float[dim*K];
        int[] prototypeIDs = new int[K];
        for (int k = 0; k < K; k++) {
            // Select Random Unique Indices for all K
            boolean randomNumberUnique = true;
                do {
                    int index = random.nextInt(n);
                    prototypeIDs[k] = index;
                    for (int j = 0; j < k; j++) {
                        if (prototypeIDs[j] == index) {
                            randomNumberUnique = false;
                        }
                    }
                } while (!randomNumberUnique);
            // Copy Raw Values into Float Array
            for (int d = 0; d <dim ; d++) {
                prototypes[k*dim+d] = data[prototypeIDs[k]+d];
            }
        }

        nearestPrototypeforVector = new int[n];
        CLContext context = JavaCL.createBestContext();

        CLQueue queue = context.createDefaultQueue();

        CLBuffer<Float> vectors = context.createFloatBuffer(Usage.Input, FloatBuffer.wrap(data), true);
        CLBuffer<Float> prototypeBuffer = context.createFloatBuffer(Usage.Input, FloatBuffer.wrap(prototypes), true);
        CLBuffer<Integer> nearestPrototype = context.createIntBuffer(Usage.InputOutput, IntBuffer.wrap(nearestPrototypeforVector), false);

//        // Read the program sources and compile them :
//        String src = IOUtils.readText(com.mycompany.main.class.getResource("TutorialKernels.cl"));
//        CLProgram program = context.createProgram(src);
//        // Get and call the kernel :
//        CLKernel addFloatsKernel = program.createKernel("find_nearest_prototype");
        long t0 = System.currentTimeMillis();

        TutorialKernels kernels = new TutorialKernels(context);
        int[] globalSizes = new int[] { n };
        CLEvent[] calculatePrototypes = new CLEvent[K];
        CLBuffer[] pointsInCluster = new CLBuffer[K];
        CLBuffer[] newprototypes = new CLBuffer[K];

        for (int t = 0; t < MAX_ITERATIONS; t++) {
            CLEvent findNearestPrototypes;
            findNearestPrototypes = kernels.find_nearest_prototype(queue, vectors, prototypeBuffer, nearestPrototype, dim, K, n, globalSizes, null);

            Pointer<Integer> outPtr = nearestPrototype.read(queue, findNearestPrototypes);
            int[] selectedPrototypes = outPtr.getInts();
            List[] buckets = new List[K];
            // Init Bucket Lists
            for (int k = 0; k < K; k++) {
                buckets[k] = new ArrayList<Integer>();
            }
            // Sort Indices into Buckets corresponding to the cluster it was close to
            try {
                for (int i = 0; i < n; i++) {
                    buckets[selectedPrototypes[i]].add(i);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            int[] globalSizesPrototypeCalculation = new int[] { 1 };
            for (int k = 0; k < K; k++) {
                int size = buckets[k].size();
                int[] intbucket = new int[size];
                for (int i = 0; i < size ; i++) {
                    intbucket[i] = (int) buckets[k].get(i);
                }
                pointsInCluster[k] = context.createIntBuffer(Usage.Input, IntBuffer.wrap(intbucket), true);
                newprototypes[k] = context.createFloatBuffer(Usage.Output, dim);
                calculatePrototypes[k] = kernels.calculate_prototype(queue,vectors,pointsInCluster[k],newprototypes[k],dim, K,size,globalSizesPrototypeCalculation,null);
            }
            for (int k = 0; k < K; k++) {
                Pointer<Float> outNewProto = newprototypes[k].read(queue, calculatePrototypes[k]);
                for (int d = 0; d < dim; d++) {
                    prototypes[k*dim +d] = outNewProto.get(d);
                }
            }

        }

        long t1 = System.currentTimeMillis();
        System.out.println(t1 - t0 + "ms");

    }

    private float distance(Vector<Float> a, Vector<Float> b){
        //Euclidian Distance - 'dim'-dimensional
        float sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += Math.pow(a.get(i) - b.get(i), 2);
        }
        return (float) Math.sqrt(sum);
    }

    private static void importCSV(String path) {

        try {
            Reader in = new FileReader(path);
            List<CSVRecord> records = CSVFormat.TDF.parse(in).getRecords();  // Tab Delimiter

            dim = records.get(0).size()-1;   // -1 ID column
            n = records.size()-1;     // -1 Header Row
            ids = new ArrayList<>(n);
            data = new float[dim*n];
            NumberFormat nf = NumberFormat.getInstance(Locale.GERMAN);

            for (int j = 1; j < n; j++) {
                CSVRecord record = records.get(j);
                ids.add(record.get(0));

                for (int i = 1; i < dim; i++) {
                    data[(j-1)*dim + i-1]= nf.parse(record.get(i)).floatValue();
                }

            }
            System.out.println();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }
    }
}
