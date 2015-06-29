package com.mycompany;

import junit.framework.TestCase;

/**
 * Created by lukas on 6/29/15.
 */
public class KmeansClusteringTest extends TestCase {

    public void setUp() throws Exception {
        super.setUp();
        String[] args = new String[]{"--numberOfClusters 5", "--path Test.csv"};
        Clustering clustering = new Clustering(args);

    }

    public void testClusteringLoop() throws Exception {

    }

    public void testCalcNewPrototypes() throws Exception {

    }

    public void testInitPrototypes() throws Exception {

    }
}