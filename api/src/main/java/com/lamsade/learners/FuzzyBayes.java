/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.lamsade.learners;

import com.fuzzylite.variable.InputVariable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A simple Naive Bayes Classifier but based on fuzzy sets.
 * 
 * 1- Start by clustering value of feature into fuzzy sets
 * 2- Compute probabilities for each fuzzy sets and labels
 * 
 * @author A. Richard
 */
public class FuzzyBayes extends MultiLabelLFCS
{
    private static final Logger LOG = LogManager.getLogger(FuzzyBayes.class);
    
    /// Map to access to input variables quickly
    private final HashMap<String, InputVariable> input_variables = new HashMap<>();
    
    /// Probabilities of each value of label
    private final HashMap<String, Double[]> label_proba = new HashMap<>();
    
    /// Probabilities of each term and label 
    private final HashMap<String, Double[]> termNlabel_proba = new HashMap<>();
    
    /// Threshold to predict label, if P(l=1|X) >= lambda -> predict l=1
    private final double lambda = 0.5;
    
    @Override
    protected void buildInternal(MultiLabelInstances mli) throws Exception {
        // Init map of input variables
        initInputVariables(mli);
        
        // Compute probabilities
        computeProbabilities(mli);
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instnc) throws Exception, InvalidDataException {
        double[] confidences = new double[numLabels];
        boolean[] predictions = new boolean[numLabels];

        LOG.debug("Get list of terms matched by the instance and compute P(X)");
        int nb_features = instnc.numAttributes() - numLabels;
        String[] matching_terms = new String[nb_features];
        for(int i = 0 ; i < nb_features ; i++){
            // Get feature fuzzy variable
            Attribute f = instnc.attribute(i);
            InputVariable v = input_variables.get(f.name());
            
            // Get feature value
            double f_value = instnc.value(f);
            
            String term = f.name() + " is ";
            term += getMatchTerm(v, f_value);
            
            // Add term into the list of matching terms
            matching_terms[i] = term;
        }
        LOG.debug("Getting list of matching terms done : " + Arrays.toString(matching_terms));
        
        LOG.debug("Start predicting value of labels");
        for(int i = 0 ; i < numLabels ; i++){
            // Get probabilities of label P(L=1) and P(L=0)
            Attribute label = instnc.attribute(nb_features + i);
            Double[] p_l = label_proba.get(label.name());
            
            // Compute P(X | L=1) and P(X | L=0)
            double[] p_xl = {1.0, 1.0};
            for(int j = 0 ; j < nb_features ; j++){
                // Get P(T|L=1) and P(T|L=0)
                String key = matching_terms[j] + " and " + label.name();
                // If no proba for key lets suppose it doesn't matter
                Double[] p_tx = {0.0, 0.0};
                if(termNlabel_proba.containsKey(key))
                    p_tx = termNlabel_proba.get(key);
                LOG.debug(key + " : " + Arrays.toString(p_tx));
                
                // Compute P(X | L=1) and P(X | L=0)
                p_xl[0] *= p_tx[0];
                p_xl[1] *= p_tx[1];
            }
            
            // Compute P(L=1|X) and P(L=0|X)
            double[] p_lx = new double[2];
            double p_x = 0.0; // Compute P(X) = P(X|L)P(L) + P(X|!L)P(!L)
            for(int k = 0 ; k < 2 ; k++){
                p_lx[k] = p_xl[k] * p_l[k];
                p_x += p_lx[k];
            }
            LOG.debug("P(" + Arrays.toString(matching_terms) + ") = " + p_lx[0] + " + " + p_lx[1] + " = " + p_x);
            
            // Compute P(L=1|X), n.b: P(L=0|X) = 1 - P(L=1|X)
            if(p_x == 0.0)
                p_x = 1.0;
            p_lx[1] = p_lx[1] / p_x;
            LOG.debug("P(" + label.name() + "=1 | " + Arrays.toString(matching_terms) + ") = (" + p_xl[1] + " * " + p_l[1] + ") / " + p_x + " = " + p_lx[1]);            
            
            // Decide of the prediction
            predictions[i] = (p_lx[1] >= this.lambda);
            confidences[i] = predictions[i] ? p_lx[1] : 1.0 - p_lx[1];
        }
        LOG.debug("Prediction for labels done");
        
        MultiLabelOutput mlo = new MultiLabelOutput(predictions, confidences);
        return mlo;
    }
    
    /**
     * Create an input variable for each feature of dataset 
     * and add it into the map of input variables.
     * 
     * @param mli Multi-label instances
     */
    protected void initInputVariables(MultiLabelInstances mli){
        LOG.debug("Start initialization of input variables");
        Instances dataset = mli.getDataSet();
        double[][] l_values = new double[numLabels][];
        int numFeatures = dataset.numAttributes() - numLabels;
        for(int l_id = 0 ; l_id < numLabels ; l_id++)
            l_values[l_id] = dataset.attributeToDoubleArray(numFeatures + l_id);
        
        for(int f_id : mli.getFeatureIndices()){
            Attribute f = dataset.attribute(f_id);
            double[] f_values = dataset.attributeToDoubleArray(f_id);
            
            LOG.debug("Create input variable for feature : {name: " + f.name() + ", nominal: " + f.isNominal() + "}");
            InputVariable f_var = createInputVariable(f, f_values, l_values);
            
            // Add input variable into hash map
            input_variables.put(f.name(), f_var);
        }
        LOG.debug("Initialization of input variables done");
    }
    
    /**
     * Compute the probability of each feature's term and label's value.
     * 
     * Use Laplace additive smoothing to avoid probabilities at zero.
     * 
     * @param mli Multi-label instances
     */
    protected void computeProbabilities(MultiLabelInstances mli){
        LOG.debug("Start counting occurences of terms and labels' values");
        // Get the number of occurences for term and label
        HashMap<String, int[]> label_occ = new HashMap<>();
        HashMap<String, int[]> termNlabel_occ = new HashMap<>();
        
        int nb_instances = mli.getNumInstances();
        Instances dataset = mli.getDataSet();
        for(int i = 0 ; i < nb_instances ; i++){
            // For each feature increment occurence of best matching term
            ArrayList<String> matching_terms = new ArrayList<>();
            for(int f_id : mli.getFeatureIndices()){
                // Get corresponding input variable
                Attribute feature = dataset.attribute(f_id);
                InputVariable f_var = input_variables.get(feature.name());
                
                // Get value
                double f_value = dataset.get(i).value(feature);
                
                // Get best matching term
                String term = feature.name() + " is ";
                term += getMatchTerm(f_var, f_value);
                
                matching_terms.add(term);
            }
            
            // For each label 
            for(int l_id : mli.getLabelIndices()){
                // Get corresponding attribute
                Attribute label = dataset.attribute(l_id);
                String l_name = label.name();
                int l_value = (int) dataset.get(i).value(label);
                
                if(!label_occ.containsKey(l_name))
                    label_occ.put(l_name, new int[2]); // label = 0�or 1
                
                // Increment occurence of current label's value
                label_occ.get(l_name)[l_value]++;
                
                // Increment occurence of current matching terms and label's value
                for(String term : matching_terms){
                    String key = term + " and " + l_name;
                    
                    if(!termNlabel_occ.containsKey(key))
                        termNlabel_occ.put(key, new int[2]); // label = 0�or 1
                    
                    termNlabel_occ.get(key)[l_value]++;
                }
            }
        }
        LOG.debug("Counting of occurences done");
        
        
        // Now we have occurences, we can compute probabilites
        LOG.debug("Compute label probabilities");
        label_proba.clear();
        for(String label : label_occ.keySet()){
            int[] occ = label_occ.get(label);
            label_proba.put(label, new Double[occ.length]);
            for(int i = 0 ; i < occ.length ; i++){
                label_proba.get(label)[i] = (double) occ[i] / (double) nb_instances;
                LOG.debug("P(" + label + " = " + i + ") = " + occ[i] + " / " + nb_instances + " = " + label_proba.get(label)[i]);
            }
        }
        
        LOG.debug("Compute (term | label) probabilities");
        termNlabel_proba.clear();
        for(String key : termNlabel_occ.keySet()){
            int[] occ = termNlabel_occ.get(key);
            termNlabel_proba.put(key, new Double[occ.length]);
            
            // Get label occurences
            String[] subkeys = key.split(" and ");
            String t_name = subkeys[0];
            String l_name = subkeys[subkeys.length - 1];
            int[] l_occ = label_occ.get(l_name);
            
            for(int i = 0 ; i < occ.length ; i++){
                termNlabel_proba.get(key)[i] = (double) occ[i] / (double) l_occ[i];
                LOG.debug("P(" + key.replace(" and ", " | ") + "=" + i + ") = " + occ[i] + " / " + l_occ[i] + " = " + termNlabel_proba.get(key)[i]);
            }
        }
        
        LOG.debug("Computing of probabilities done");
    }
}
