/*
 * This file is part of ML-LFCS.
 *
 * ML-LFCS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ML-LFCS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ML-LFCS.  If not, see <https://www.gnu.org/licenses/>.
 */
package com.lamsade.learners;


import com.fuzzylite.Engine;
import com.fuzzylite.activation.General;
import com.fuzzylite.defuzzifier.Centroid;
import com.fuzzylite.norm.s.Maximum;
import com.fuzzylite.norm.t.Minimum;
import com.fuzzylite.rule.Rule;
import com.fuzzylite.rule.RuleBlock;
import com.fuzzylite.term.Discrete;
import com.fuzzylite.term.Discrete.Pair;
import com.fuzzylite.term.Ramp;
import com.fuzzylite.term.Term;
import com.fuzzylite.term.Triangle;
import com.fuzzylite.variable.InputVariable;
import com.fuzzylite.variable.OutputVariable;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.data.MultiLabelInstances;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 * Implementation of the multi-label Learning Fuzzy Classifier System.
 * 
 * This classifier is base on two part:
 * 1 - Clustering variable values using Fuzzy c-means
 * 2 - Build simple rules sets with Q = P(H|X)
 * 
 * @author A. Richard
 */
public abstract class MultiLabelLFCS extends MultiLabelLearnerBase
{
    private static final Logger LOG = LogManager.getLogger(MultiLabelLFCS.class);
    
    /// Internal fuzzy engine
    private static final Engine ENGINE = new Engine("ml-lfcs");
    
    /**
     * Create and return a input variable corresponding to a feature attribute.
     * 
     * Membership Functions of the input variable are computed
     * according to the type of the feature attribute:
     * - Nominal: Ramp or Triangle function of each possible value.
     * - Numeric: Clustering of values using Fuzzy c-means algorithm
     * 
     * @param a Feature attribute corresponding to the input variable
     * @param a_values set of attribute's values for each instance
     * @param l_values matrix of labels' values
     * 
     * @return An InputVariable
     */
    protected InputVariable createInputVariable(
            Attribute a, 
            double[] a_values,
            double[][] l_values
        )
    {
        String name = a.name();
        Double min = (a.isNumeric() ? 
                a.getLowerNumericBound() 
                : // else
                (a.isString() ?
                1.0
                :
                0.0
                )
            );
        Double max = (a.isNumeric() ? 
                a.getUpperNumericBound() 
                : // else
                a.numValues()-1
            );
        
        InputVariable v = new InputVariable(name, min, max);
        v.setEnabled(true);
        v.setLockValueInRange(false);
        
        // init terms for nominal attributes
        if(a.isNominal()){
            for(int i = 0 ; i < a.numValues() ; i++){
                if(i==0)
                    v.addTerm(new Ramp(a.value(i), i+1, i)); // Ramp(i) = 1 and Ramp(i+1) = 0
                else if(i==a.numValues()-1)
                    v.addTerm(new Ramp(a.value(i), i-1, i)); // Ramp(i-1) = 0 and Ramp(i) = 1 
                else
                    v.addTerm(new Triangle(a.value(i), i-1, i, i+1));
            }
        }
        else if (a.isNumeric()){
            LOG.debug("Start clustering of values of " + a.name() + ": " + Arrays.toString(a_values));
            Discrete[] terms = fuzzy_c_means(a_values, 5, 2, 0.01);
            
            for(Discrete term : terms)
                v.addTerm(term);
            
            if(LOG.isDebugEnabled())
                printDiscretesToCsv(terms, "../tmp/"+a.name()+".csv");
        }
        
        LOG.debug("Input variable created:\n" + v.toString());
        
        return v;
    }
    
    /**
     * Create and return an output variable corresponding to a label attribute.
     * 
     * Membership functions of the output variable are Ramp or Triangle
     * corresponding to each possible value of label attribute.
     * 
     * @param a label attribute
     * 
     * @return An ouput variable corresponding to a label
     */
    protected OutputVariable createOutputVariable(Attribute a){        
        OutputVariable v = new OutputVariable(a.name(), 0, a.numValues()-1);
        v.setEnabled(true);
        v.setLockValueInRange(false);
        v.setLockPreviousValue(false);
        v.setDefaultValue(0);
        
        v.setAggregation(new Maximum());
        v.setDefuzzifier(new Centroid());
        
        // create terms (can be ranking)
        for(int i = 0 ; i < a.numValues() ; i++){
            if(i==0)
                v.addTerm(new Ramp(String.valueOf(i), i, i+1));
            else if(i==a.numValues()-1)
                v.addTerm(new Ramp(String.valueOf(i), i-1, i));
            else
                v.addTerm(new Triangle(String.valueOf(i), i-1, i, i+1));
        }
        
        LOG.debug("Output Variable created:\n" + v.toString());
        
        return v;
    }
    
    /**
     * Based on dataset and fuzzy engine's variables this function creates
     * a set of assumptions rules for internal fuzzy engine.
     * 
     * @param mli Multi-labeled instances
     */
    protected void createAssumptionRules(MultiLabelInstances mli){
        LOG.debug("Start parsing possible assumptions");
        HashMap<String, Integer> ant_occs = new HashMap<>();
        HashMap<String, Integer> cons_occs = new HashMap<>();
        HashMap<String, Integer> ant_cons_occs = new HashMap<>();
        // Check for possible assumptions O(N*(T + L))
        Set<String> assumptions = getPossibleAssumptions(mli, ant_occs, cons_occs, ant_cons_occs);
        LOG.debug("Parsing of possible assumptions done");
        
        // Get sets of rules for each label
        double nb_instnc = (double) mli.getNumInstances();
        for(Attribute a : mli.getLabelAttributes()){
            RuleBlock rb = createRuleBlock(
                    a.name(), 
                    "Implication hypothesis for label " + a.name()
                );
            LOG.debug(rb.getDescription());
            
            for(String assumption : assumptions){
                if(assumption.contains(a.name())){
                    Rule rule = Rule.parse(assumption, ENGINE);
                    
                    
                    // Get occurence of the antecedent
                    String antecedent = rule.getAntecedent().toString();
                    antecedent = antecedent.substring(0, antecedent.length()-1);
                    double ant_occ = (double) ant_occs.get(antecedent);
                    // Compute probability of antecedent
                    double p_ant = ant_occ / nb_instnc;
                    LOG.trace("P(T) = " + ant_occ + " / " + nb_instnc + " = " + p_ant);
                    
                    // Get occurence of the consequent
                    String consequent = rule.getConsequent().toString();
                    double cons_occ = (double) cons_occs.get(consequent);
                    // Compute probability of the consequent
                    double p_cons = cons_occ / nb_instnc;
                    LOG.trace("P(L) = " + cons_occ + " / " + nb_instnc + " = " + p_cons);
                    
                    LOG.trace("P(L)*P(T) = " + (p_ant*p_cons));
                    
                    // Get occurence of antecedent and consequent
                    double ant_cons_occ = (double) ant_cons_occs.get(assumption);
                    // Compute probability of antecedent and consequent
                    double p_ant_cons = ant_cons_occ / nb_instnc;
                    LOG.trace("P(T and L) = " + ant_cons_occ + " / " + nb_instnc + " = " + (ant_cons_occ / nb_instnc));
                    
                    
                    // Compute P(ant | cons) = P(ant and cons) / P(cons)
                    double p_xh = p_ant_cons / p_cons;
                    LOG.trace("P(T|L) = " + p_ant_cons + " / " + p_cons + " = " + p_xh);
                    
                    // Compute P(cons | ant) = P(ant and cons) / P(ant)
                    double p_hx = p_ant_cons / p_ant;
                    LOG.trace("P(L|T) = " + p_ant_cons + " / " + p_ant + " = " + p_hx);
                    
                    rule.setWeight(p_hx);
                    rb.addRule(rule);
                    LOG.debug(rule.toString() + " - weight : " + rule.getWeight());
                    
                    
                }
            }
            
            ENGINE.addRuleBlock(rb);
        }
        LOG.debug("Creation of assumption rules done:\n" + ENGINE.toString());
    }
    
    /**
     * Parse dataset and compute possible assumptions.
     * 
     * A�rule is created for each different combination of Features values
     * when a label is true
     * 
     * @param mli dataset
     * @param antecedent_occurences will contain number of occurences of each antecedent
     * @param consequent_occurences will contain number of occurences of each consequent
     * @param ant_cons_occurences will contain number of occurences of each antecedent and consequent
     * 
     * @return A Set of possible assumptions
     */
    protected HashSet<String> getPossibleAssumptions(
            MultiLabelInstances mli,
            HashMap<String, Integer> antecedent_occurences,
            HashMap<String, Integer> consequent_occurences,
            HashMap<String, Integer> ant_cons_occurences
        )
    {
        HashSet<String> rules = new HashSet<>();
        Instances dataset = mli.getDataSet();
        for(int i = 0 ; i < dataset.numInstances() ; i++){
            // Compute antecedent
            String antecedent = "";
            boolean first = true;
            for(int f_id : mli.getFeatureIndices()){
                antecedent += (first ? "" : " and ");
                if(first) first = false;
                
                String f_name = dataset.attribute(f_id).name();
                double f_value = dataset.attributeToDoubleArray(f_id)[i];
                
                String match_term = getMatchTerm(f_name, f_value);
                
                antecedent += f_name + " is " + match_term;                
            }
            
            // Update number of occurence for this antecedent
            if(antecedent_occurences.containsKey(antecedent)){
                int occ = antecedent_occurences.get(antecedent);
                antecedent_occurences.put(antecedent, occ + 1);
            }
            else
                antecedent_occurences.put(antecedent, 1);
            
            // Make assumption for each true label
            for(int l_id : mli.getLabelIndices()){
                int l_value = (int) dataset.attributeToDoubleArray(l_id)[i];
                
                if(l_value > 0){
                    String l_name = dataset.attribute(l_id).name();
                    String consequent = l_name + " is " + l_value;
                    
                    // Update number of occurence for this consequent
                    if(consequent_occurences.containsKey(consequent)){
                        int occ = consequent_occurences.get(consequent);
                        consequent_occurences.put(consequent, occ + 1);
                    }
                    else
                        consequent_occurences.put(consequent, 1);
                    
                    String rule = "if " + antecedent + " then " + consequent;
                    rules.add(rule);
                    
                    // Update number of occurence for this rule
                    if(ant_cons_occurences.containsKey(rule)){
                        int occ = ant_cons_occurences.get(rule);
                        ant_cons_occurences.put(rule, occ + 1);
                    }
                    else
                        ant_cons_occurences.put(rule, 1);
                }
            }
        }
        
        return rules;
    }
    
    /**
     * Get the best match term for a given feature value.
     * 
     * @param feature_name name of the feature variable
     * @param feature_value the value to check
     * 
     * @return the name of the best matching term
     */
    protected String getMatchTerm(String feature_name, double feature_value){

        InputVariable v = ENGINE.getInputVariable(feature_name);
        String best_match_name = "";
        double best_match = Double.MIN_VALUE;
        for(Term t : v.getTerms()){
            double match = t.membership(feature_value); 
            if(match > best_match){
                best_match = match;
                best_match_name = t.getName();
            }
        }
        
        return best_match_name;
    }
    
    /**
     * Get the best match term for a given feature value.
     * 
     * @param feature_var the feature variable
     * @param feature_value the value to check
     * 
     * @return the name of the best matching term
     */
    protected String getMatchTerm(InputVariable feature_var, double feature_value){

        String best_match_name = "";
        double best_match = Double.MIN_VALUE;
        for(Term t : feature_var.getTerms()){
            double match = t.membership(feature_value); 
            LOG.debug(feature_var.getName() + " is " + t.getName() + ": mu(" + feature_value + ") = " + match);
            if(match > best_match){
                best_match = match;
                best_match_name = t.getName();
            }
        }
        
        return best_match_name;
    }
    
    /**
     * Create ande return a rule block.
     * 
     * Conjunction : min
     * Disjunction : max
     * Implicatino : AlgebraicProcduct
     * Activation : General
     * 
     * @param name name of the rule block to create
     * @param description description of the rule block to create
     * 
     * @return a new RuleBlock
     */
    protected RuleBlock createRuleBlock(String name, String description){
        
        RuleBlock rb = new RuleBlock();
        rb.setName(name);
        rb.setDescription(description);
        rb.setEnabled(true);
        
        rb.setConjunction(new Minimum());
        rb.setDisjunction(new Maximum());
        rb.setImplication(new Minimum());
        rb.setActivation(new General());
        
        return rb;
    }
    
    /**
     * Fuzzy c-means algorithm.
     * 
     * Cluster a set of values into c clusters corresponding
     * to c discrete membership fonctions.
     * 
     * @param values values to cluster
     * @param nb_cluster number of clusters (2 ≤ c ≤ N)
     * @param cluster_fuzzyness fuzzyness parameter (1 < m)
     * @param epsilon parameter to stop clustering
     * 
     * @return a set of discrete membership functions
     */
    protected Discrete[] fuzzy_c_means(
            double[] values, 
            int nb_cluster, 
            int cluster_fuzzyness, 
            double epsilon
        )
    {
        // Build sorted copy without duplicates
        double[] cpy = sortedCopyWithoutDuplicates(values);
        LOG.debug("Sort values and remove duplicates: " + Arrays.toString(cpy));
        
        // also build a sorted values array
        double[] sorted_values = Arrays.copyOf(values, values.length);
        Arrays.sort(sorted_values);
        
        LOG.debug("Initialize matrix U_0");
        // If there is less n distinct values than c clusters do n clusters 
        int c = cpy.length < nb_cluster ? cpy.length : nb_cluster;
        double[][] U = new double[cpy.length][c];
        double[] C = new double[c]; // centroids
        
        // launch clustering
        int step = 0;
        double max_dist;
        LOG.debug("Start fuzzy c-means main loop until max_ij(u_ij^k+1 - u_ij^k) < "+epsilon);
        do{
            if(step == 0){ // Initialisation
                // choose centroids form sorted values
                double min = cpy[0];
                double max = cpy[cpy.length-1];
                for(int j = 0 ; j < c ; j++){
                    if(c == cpy.length)
                        C[j] = cpy[j];
                    else if(j == 0)
                        C[j] = min;
                    else if(j == c-1)
                        C[j] = max;
                    else 
                        C[j] = min + j*((max - min) / (c - 1));
                }
            }
            else{
                // Compute new centroids based on values array
                for(int j = 0 ; j < c ; j++){
                    double u_sum = 0.0;
                    double x_sum = 0.0;
                    for(int v = 0 ; v < sorted_values.length ; v++){
                        double x = sorted_values[v];
                        int i = Arrays.binarySearch(cpy, x);
                        
                        double um = Math.pow(U[i][j], cluster_fuzzyness);
                        u_sum += um;
                        x_sum += um*x;
                    }

                    if(u_sum != 0.0)
                        C[j] = x_sum / u_sum;
                }
            }
            LOG.debug("Centroids for step "+step+" : "+Arrays.toString(C));
            
            // Update U
            max_dist = step == 0 ? Double.MAX_VALUE : Double.MIN_VALUE;
            for(int i = 0 ; i < cpy.length ; i++){ // for each value i
            
                for(int j = 0 ; j < c ; j++){ // for each cluster j

                    // compute distance of x_i from c_j
                    double abs_dist = Math.abs(cpy[i] - C[j]);

                    // compute sum
                    double u = 0.0;
                    if(abs_dist == 0.0) // if x_i is the centroid c_j
                        u = 1.0;
                    else{
                        for(int k = 0 ; k < c ; k++){
                            double tmp = abs_dist / Math.abs(cpy[i] - C[k]);
                            u += Math.pow(tmp, 2/(cluster_fuzzyness-1));
                        }
                    }

                    if(step != 0){ // If not initialization step
                        // Check dist from U_k
                        double dist = Math.abs(1/u - U[i][j]);
                        if(dist > max_dist)
                            max_dist = dist;
                    }
                    
                    // Set u_ij as the inverse of sum
                    U[i][j] = 1/u;
                }
            }
            
            LOG.debug("U_"+step+": "+Arrays.deepToString(U)+" (||U_k+1 - U_k|| = " + max_dist+")");
            
            step++;
            
        }while(max_dist >= epsilon);

        LOG.debug("Fuzzy c-means done with "+(step-1)+" step and "+max_dist+" dist");
        
        // Finally, return an array of discrete fuzzy terms
        return cmeans2discretes(U, C, c, cpy);
    }
    
    /**
     * Variation of fuzzy c-means algorithm to use labels' values to build
     * clusters.
     * 
     * @param values feature's values
     * @param l_values labels' values
     * @param nb_cluster maximum of clusters
     * @param cluster_fuzzyness fuzzyness of the clusters
     * @param epsilon precision degree 
     * 
     * @return An array of discrete fuzzy terms
     */
    protected Discrete[] fuzzy_c_means(
            double[] values,
            double[][] l_values,
            int nb_cluster, 
            int cluster_fuzzyness, 
            double epsilon
        )
    {
        
        LOG.debug("Initialize matrix U_0 and centroids");
        // If there is less n values than c clusters do n clusters 
        int c = values.length < nb_cluster ? values.length : nb_cluster;
        double[][] U = new double[values.length][c];
        double[][] C = new double[c][numLabels+1]; // centroids x + set of labels
        
        // launch clustering
        int step = 0;
        double max_dist;
        LOG.debug("Start fuzzy c-means main loop until max_ij(u_ij^k+1 - u_ij^k) < "+epsilon);
        do{
            if(step == 0){ // Initialisation
                double[] cpy = Arrays.copyOf(values, values.length);
                Arrays.sort(cpy);
                // choose centroids form sorted values
                double min = cpy[0];
                double max = cpy[cpy.length-1];
                for(int j = 0 ; j < c ; j++){
                    if(c == cpy.length)
                        C[j][0] = cpy[j]; // labels at 0.0 by default
                    else if (j == 0)
                        C[j][0] = min;
                    else if (j == c-1)
                        C[j][0] = max;
                    else 
                        C[j][0] = min + j*((max - min) / (c - 1));
                }
            }
            else{
                // Compute new centroids based on values array
                for(int j = 0 ; j < c ; j++){
                    double u_sum = 0.0;
                    double[] x_sum = new double[numLabels+1];
                    for(int i = 0 ; i < values.length ; i++){
                        double um = Math.pow(U[i][j], cluster_fuzzyness);
                        u_sum += um;
                        
                        double[] x = new double[numLabels+1];
                        for(int l = 0 ; l < numLabels+1 ; l++){ // get labels of x
                            x[l] = l== 0 ? values[i] : l_values[l-1][i];
                            x_sum[l] += um*x[l];
                        }
                    }

                    if(u_sum != 0.0){
                        for(int i = 0 ; i < numLabels+1 ; i++){
                            C[j][i] = x_sum[i] / u_sum;
                        }
                    }
                }
            }
            LOG.debug("Centroids for step "+step+" : "+Arrays.deepToString(C));
            
            // Update U
            max_dist = step == 0 ? Double.MAX_VALUE : Double.MIN_VALUE;
            for(int i = 0 ; i < values.length ; i++){ // for each value i
            
                for(int j = 0 ; j < c ; j++){ // for each cluster j

                    // compute distance of x_i from c_j
                    double[] x = new double[numLabels+1];
                    x[0] = values[i];
                    for(int l = 0 ; l < numLabels ; l++)
                        x[l+1] = l_values[l][i];
                    double dist = manhattanDistance(x, C[j]);

                    // compute sum
                    double u = 0.0;
                    if(dist == 0.0) // if x_i is the centroid c_j
                        u = 1.0;
                    else{
                        for(int k = 0 ; k < c ; k++){
                            double tmp = dist / manhattanDistance(x, C[k]);
                            u += Math.pow(tmp, 2/(cluster_fuzzyness-1));
                        }
                    }
                    u = 1/u;

                    if(step != 0){ // If not initialization step
                        // Check dist from U_k
                        double n_dist = Math.abs(u - U[i][j]);
                        if(n_dist > max_dist)
                            max_dist = n_dist;
                    }
                    
                    // Set u_ij as the inverse of sum
                    U[i][j] = u;
                }
            }
            
            LOG.debug("U_"+step+": "+Arrays.deepToString(U)+" (||U_k+1 - U_k|| = " + max_dist+")");
            
            step++;
            
        }while(c != values.length && max_dist >= epsilon);

        LOG.debug("Fuzzy c-means done with "+(step-1)+" step and "+max_dist+" dist");
        
        double[] CC = new double[c];
        for(int j = 0 ; j < c ; j++)
            CC[j] = C[j][0];
        return cmeans2discretes(U, CC, c, values);
    }
    
    /**
     * Transform results of fuzzy c-means into discrete fuzzy terms
     * 
     * @param U the matrix of degrees U
     * @param C the array of centroids
     * @param c the number of cluster
     * @param v the array of distinct values
     * 
     * @return An array of Discrete fuzzy terms
     */
    protected Discrete[] cmeans2discretes(
            double[][] U,
            double[] C,
            int c,
            double[] v
        )
    {
        Discrete[] terms = new Discrete[c];
        for(int j = 0 ; j < c ; j++){
            boolean[] centroid_added = new boolean[c];
            List<Pair> pairs = new ArrayList<>();
            for(int i = 0 ; i < v.length ; i++){
                double x = v[i];
                for(int k = 0 ; k < c ; k++){ // Centroids too
                    if(!centroid_added[k] && x > C[k]){
                        pairs.add(new Pair(C[k], (k == j ? 1.0 : 0.0)));
                        centroid_added[k] = true;
                    }
                }
                pairs.add(new Pair(x, U[i][j]));
            }
            
            Discrete.sort(pairs);
            terms[j] = new Discrete("u"+String.valueOf(j), pairs);
        }
        
        return terms;
    }
    
    /**
     * Compute and return the manhattan distance between two points
     * in n dimensions.
     * 
     * @param p1 first point
     * @param p2 second point
     * 
     * @return the manhattan distance between p1 and p2
     */
    protected double manhattanDistance(double[] p1, double p2[]){
        if(p1.length != p2.length){
            LOG.error("the number of dimensions must be same (p1 length=" + p1.length + ", p2 length=" + p2.length);
            return -1.0;
        }
        
        double sum = 0.0;
        for(int i = 0 ; i < p1.length ; i++)
            sum += Math.abs(p1[i] - p2[i]);
        
        return sum;
    }
    
    protected double[] sortedCopyWithoutDuplicates(double[] values){
        double[] cpy = Arrays.copyOf(values, values.length);
        Double[] obj_cpy = ArrayUtils.toObject(cpy);
        TreeSet<Double> tree = new TreeSet<>(Arrays.asList(obj_cpy));
        obj_cpy = tree.toArray(new Double[0]);
        cpy = ArrayUtils.toPrimitive(obj_cpy);
        
        return cpy;
    }
    
    /**
     * Compute and return the euclidian distance between two points
     * in n dimensions.
     * 
     * @param p1 first point
     * @param p2 second point
     * 
     * @return the euclidian distance between p1 and p2
     */
    protected double euclidianDistance(double[] p1, double p2[]){
        if(p1.length != p2.length){
            LOG.error("the number of dimensions must be same (p1 length=" + p1.length + ", p2 length=" + p2.length);
            return -1.0;
        }
        
        double sum = 0.0;
        for(int i = 0 ; i < p1.length ; i++)
            sum += Math.pow(p1[i] - p2[i], 2);
        
        return Math.sqrt(sum);
    }
    
    /**
     * Print array of discretes terms of variable to plot result of fuzzy c-means.
     * 
     * @param terms
     * @param csv_path 
     */
    private void printDiscretesToCsv(Discrete[] terms, String csv_path){
        try{
            // Init ouput file
            File f = new File(csv_path);
            if(!f.exists())
                f.getParentFile().mkdirs();
            else
                f.delete();
            f.createNewFile();
            
            // Write headers
            String output = "x";
            for(int i = 0 ; i < terms.length ; i++)
                output += ";u"+String.valueOf(i);
            output += ";cluster\n";
            
            // Parse values
            int nb_values = terms[0].getXY().size();
            for(int i = 0 ; i < nb_values ; i++){
                output += String.valueOf(terms[0].get(i).x);
                int c = 0; // best match cluster
                for(int j = 0 ; j < terms.length ; j++){
                    output += ";"+String.valueOf(terms[j].get(i).y);
                    if(terms[j].get(i).y > terms[c].get(i).y) // if better match
                        c = j;
                }
                output += ";"+String.valueOf(c)+"\n";
            }
            
            // Print output in csv file
            PrintWriter writer = new PrintWriter(f.getPath(), "UTF-8");
            writer.print(output);
            writer.flush();
        }catch(Exception e){
            LOG.error(e);
        }
    }
    
    @Override
    public TechnicalInformation getTechnicalInformation() {
        return new TechnicalInformation(TechnicalInformation.Type.PHDTHESIS);
    }
}
