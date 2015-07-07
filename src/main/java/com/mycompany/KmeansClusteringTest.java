package com.mycompany;

import junit.framework.TestCase;

import static java.lang.Math.random;


/**
 * Created by lukas on 6/29/15.
 */
public class KmeansClusteringTest extends TestCase {

    KmeansClustering kmeansClustering;
    int k;
    int n;
    int dim;
    float[] testdata;

    public void setUp() throws Exception {
        super.setUp();
    }

    private void initSimpleTestSet1Dim() {
        k = 2;
        n = 100;
        dim = 1;
        testdata = new float[n];
        for (int i = 0; i < n / 2; i++) {
            testdata[i] = (float) random() + 3;
        }
        for (int i = n / 2; i < n; i++) {
            testdata[i] = (float) random() - 3;
        }

        kmeansClustering = new KmeansClustering(k, dim, testdata, false);
    }

    public void testClusteringLoop() throws Exception {
        initSimpleTestSet1Dim();
        int[] assignments = kmeansClustering.clusteringLoop();
        assertEquals(n, assignments.length);
        int[] count = new int[k];
        for (int i = 0; i < n; i++) {
            count[assignments[i]]++;
        }
        for (int i = 0; i < k; i++) {
            assertTrue(count[i] > 0);
        }

    }

    public void testCalcNewPrototypes() throws Exception {
        float[] test = new float[]{3, 4, -3, -4};
        float[] oldprotos = new float[]{4, -4};
        float[] newprotos = kmeansClustering.calcNewPrototypes(new int[]{0, 0, 1, 1}, 2, 4, 1, test, oldprotos);
        assertTrue(newprotos[0] < oldprotos[0]);
        assertTrue(newprotos[1] > oldprotos[1]);
    }

    public void testInitPrototypes() throws Exception {
        float[] initPrototypes = kmeansClustering.initPrototypes();
        assertEquals(k * dim, initPrototypes.length);
        for (int i = 0; i < initPrototypes.length; i++) {
            float datum = initPrototypes[i];
            assertNotNull(datum);
        }
    }
}