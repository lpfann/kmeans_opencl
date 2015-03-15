package com.mycompany;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.util.*;
import com.nativelibs4java.util.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.bridj.Pointer;
import static org.bridj.Pointer.*;
import static java.lang.Math.*;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class main {
    static Float[][] data = new Float[0][];
    static ArrayList<String> ids;
    public static final int K = 10;

    public static void main(String[] args) throws IOException {
        String path = "C:\\Users\\mirek_000\\Documents\\Dropbox\\workspace\\Clustering_Project\\GEOdata_Cholesteatom_nurLogRatio.csv";
        importCSV(path);

        CLContext context = JavaCL.createBestContext();
        CLQueue queue = context.createDefaultQueue();

        int n = 1024;
        
        // Create OpenCL input and output buffers
        CLBuffer<Float>
            a = context.createFloatBuffer(Usage.InputOutput, n), // a and b and read AND written to
            b = context.createFloatBuffer(Usage.InputOutput, n),
            out = context.createFloatBuffer(Usage.Output, n);

        TutorialKernels kernels = new TutorialKernels(context);
        int[] globalSizes = new int[] { n };
        CLEvent fillEvt = kernels.fill_in_values(queue, a, b, n, globalSizes, null);
        CLEvent addEvt = kernels.add_floats(queue, a, b, out, n, globalSizes, null, fillEvt);
        
        Pointer<Float> outPtr = out.read(queue, addEvt); // blocks until add_floats finished

        // Print the first 10 output values :
        for (int i = 0; i < 10 && i < n; i++)
            System.out.println("out[" + i + "] = " + outPtr.get(i));
        
    }

    private static void importCSV(String path) {

        try {
            Reader in = new FileReader(path);
            List<CSVRecord> records = CSVFormat.TDF.parse(in).getRecords();  // Tab Delimiter

            int dimensions = records.get(0).size();
            int n = records.size();
            ids = new ArrayList<>();
            data = new Float[dimensions][n];
            NumberFormat nf = NumberFormat.getInstance(Locale.GERMAN);

            for (int j = 1; j < n; j++) {
                CSVRecord record = records.get(j);
                ids.add(record.get(0));
                for (int i = 1; i < dimensions; i++) {
                    data[i - 1][j] = nf.parse(record.get(i)).floatValue();
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }
    }
}
