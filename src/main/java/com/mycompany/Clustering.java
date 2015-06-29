package com.mycompany;

import org.apache.commons.cli.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Created by lukas on 6/29/15.
 */
public class Clustering {
    private static int K;
    private static boolean sout;
    private int[] assignments;
    private int dim;
    private int n;
    private ArrayList<String> ids;
    private float[] data;

    public Clustering(String[] args) {
        String path = parseArgs(args);
        importCSV(path);
        KmeansClustering kmeans = new KmeansClustering(K, dim, data, false);
        assignments = kmeans.clusteringLoop();

    }

    public static void main(String[] args) throws IOException {

        Clustering clustering = new Clustering(args);


        if (sout) {
            clustering.printAssignments();
        } else {
            clustering.writeFile();
        }

        //       long t0 = System.currentTimeMillis();


        //       long t1 = System.currentTimeMillis();
        //       System.out.println(t1 - t0 + "ms");
    }

    public void printAssignments() {
        for (int i = 0; i < n; i++) {
            System.out.println(assignments[i]);
        }

    }

    private String parseArgs(String[] args) {
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

        } catch (ParseException e) {
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
     * @throws IOException
     */
    public void writeFile() throws IOException {
        String filename = "out.dat";
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(filename));
        for (int i = 0; i < n; i++) {
            bufferedWriter.write(assignments[i] + "");
            bufferedWriter.newLine();
        }
        bufferedWriter.flush();
        bufferedWriter.close();
    }

    /**
     * Import CSV Files and store it in data array
     *
     * @param path
     */
    public float[] importCSV(String path) {

        try {
            Reader in = new FileReader(path);
            // CSVFormat.TDF for Tabs or CSVFormat.DEFAULT for Commas
            List<CSVRecord> records = CSVFormat.TDF.parse(in).getRecords();  // Tab Delimiter

            dim = records.get(0).size() - 1;   // -1 ID column
            n = records.size() - 1;     // -1 Header Row
            ids = new ArrayList<>(n); // Store Names for each Point
            data = new float[dim * n];
            NumberFormat nf = NumberFormat.getInstance(Locale.GERMAN);

            for (int j = 1; j < n + 1; j++) {
                CSVRecord record = records.get(j);
                ids.add(record.get(0));

                for (int i = 1; i < dim + 1; i++) {
                    data[(j - 1) * dim + i - 1] = nf.parse(record.get(i)).floatValue();
                }

            }
            return data;
        } catch (IOException | java.text.ParseException e) {
            e.printStackTrace();
            System.out.println("Loading of InputFile was not possible. Exiting...");
            System.exit(-1);

        }
        return null;
    }
}
