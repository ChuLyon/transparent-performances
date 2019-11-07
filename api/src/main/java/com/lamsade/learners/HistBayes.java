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

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 * This class implements a variation of the Naive Bayes algorithm.
 * 
 * 1- Discretizing numerical features into histograms
 * 2- Compute frequencies of occurences
 * 
 * @author A. Richard
 */
public class HistBayes extends MultiLabelLearnerBase{
    
    private static final Logger LOG = LogManager.getLogger(HistBayes.class);
    
    /// Histograms properties for each numerical variable
    private final Map<String, double[]> histograms = new HashMap<>();
    
    private final Map<String, List<String>> nominal_values = new HashMap<>();
    
    /// Probabilities of each value of label
    private final HashMap<String, Double[]> label_proba = new HashMap<>();
    
    /// Probabilities of each term and label 
    private final HashMap<String, Double[]> termNlabel_proba = new HashMap<>();
    
    /// Threshold to predict label, if P(l=1|X) >= lambda -> predict l=1
    private final double lambda = 0.5;
    
    /// Number of bars by histograms
    private final int n = 5;

    @Override
    protected void buildInternal(MultiLabelInstances mli) throws Exception {
        initHistograms(mli);
        
        computeProbabilities(mli);
    }
    
    /**
     * Init properties of histograms for numerical features.
     * 
     * @param mli 
     */
    protected void initHistograms(MultiLabelInstances mli){
        LOG.debug("Start discretize numerical variables ...");
        Instances inst = mli.getDataSet();
        for(int i : mli.getFeatureIndices()){
            Attribute a = inst.attribute(i);
            if(a.isNumeric()){
                if(!histograms.containsKey(a.name())){
                    double[] values = inst.attributeToDoubleArray(i);

                    // cpy values (just in case) and sort values
                    double[] cpy = Arrays.copyOf(values, values.length);
                    Arrays.sort(cpy);

                    double min = cpy[0];
                    double max = cpy[cpy.length - 1];
                    double h = (max - min) / ((double) n);

                    LOG.debug("Feature "  
                            + a.name()
                            + " into " + n + " histograms: {"
                            + "min=" + min + ", "
                            + "max=" + max + ", "
                            + "h=" + h
                            + "}"
                    );

                    // Add histogram properties min, max and h into histogram map
                    histograms.put(a.name(), new double[]{min,max,h});
                   
                }
            }
            else{ // nominal features
                // Remember values of nominal features
                if(!nominal_values.containsKey(a.name())){
                    ArrayList<String> values = new ArrayList<>();
                    for(int j = 0 ; j < a.numValues() ; j++)
                        values.add(a.value(j));
                    
                    nominal_values.put(a.name(), values);
                }
            }
        }
        LOG.debug("Discretization done.");
    }
    
    protected void computeProbabilities(MultiLabelInstances mli){
        LOG.debug("Start counting occurences of features' values and labels' values");
        // Get the number of occurences for term and label
        HashMap<String, int[]> label_occ = new HashMap<>();
        HashMap<String, int[]> termNlabel_occ = new HashMap<>();
        
        int nb_instances = mli.getNumInstances();
        Instances dataset = mli.getDataSet();
        for(int i = 0 ; i < nb_instances ; i++){
            
            // For each label, increase occurence number
            for(int l_id : mli.getLabelIndices()){
                // Get corresponding attribute
                Attribute label = dataset.attribute(l_id);
                String l_name = label.name();
                int l_value = (int) dataset.get(i).value(label);
                
                if(!label_occ.containsKey(l_name))
                    label_occ.put(l_name, new int[2]); // label = 0 or 1
                
                // Increment occurence of current label's value
                label_occ.get(l_name)[l_value]++;
                
                // Increment occurence of current matching terms and label's value
                for(int f_id : mli.getFeatureIndices()){
                    Attribute feature = dataset.attribute(f_id);
                    double f_value = dataset.get(i).value(feature);
                    
                    String term = getMatchingTerm(feature, f_value);
                        
                    
                    String key = term + " and " + l_name;
                    
                    if(!termNlabel_occ.containsKey(key))
                        termNlabel_occ.put(key, new int[2]); // label = 0 or 1
                    
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
    
    /**
     * Compute the term/subset of a feature corresponding to a specific value
     * 
     * @param feature
     * @param feature_value
     * 
     * @return the matching term
     */
    protected String getMatchingTerm(Attribute feature, double feature_value){
        String term = feature.name();
        if(feature.isNumeric()){
            String h_range = getHistogramRange(term, feature_value);

            term += " in " + h_range;
        }
        else // nominal feature
            term += "=" + feature.value((int) feature_value);
        
        return term;
    }
    
    protected String getHistogramRange(String feature_name, double feature_value){
        if(!histograms.containsKey(feature_name))
            return null;
        
        // Get histogram property
        double[] h_properties = histograms.get(feature_name);
        double min = h_properties[0];
        double max = h_properties[1];
        double h = h_properties[2];

        // Get corresponding subset
        int id = n-1;
        if(feature_value != max)
            id = (int) ((feature_value - min) / h);

        return "[" + (min + id*h) + ", " + (min + (id+1)*h) + "[";
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instnc) throws Exception, InvalidDataException {
        double[] confidences = new double[numLabels];
        boolean[] predictions = new boolean[numLabels];

        LOG.debug("Get list of terms matched by the instance and compute P(X)");
        int nb_features = instnc.numAttributes() - numLabels;
        String[] matching_terms = new String[nb_features];
        for(int i = 0 ; i < nb_features ; i++){
            // Get feature and feature value
            Attribute f = instnc.attribute(i);
            double f_value = instnc.value(f);
            
            String term = getMatchingTerm(f, f_value);
            
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
        LOG.debug("Prediction for labels done: " + Arrays.toString(predictions) + " (confidences: " + Arrays.toString(confidences) + ")");
        
        MultiLabelOutput mlo = new MultiLabelOutput(predictions, confidences);
        return mlo;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        return new TechnicalInformation(TechnicalInformation.Type.PHDTHESIS);
    }
    
    /**
     * Save classifier learned into xml file.
     * 
     * This function is quite ugly, to do better.
     * 
     * @param file_path path to the xml file
     */
    public void saveToFile(String file_path){
            
        try{
            DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder docBuilder = docFactory.newDocumentBuilder();

            // root elements
            Document doc = docBuilder.newDocument();
            Element rootElement = doc.createElement("classifier");
            doc.appendChild(rootElement);
            
            HashSet<String> taboo = new HashSet<>(); 
            
            // Add Features node
            Element features = doc.createElement("features");
            rootElement.appendChild(features);
            
            // Add Labels node
            Element labels = doc.createElement("labels");
            rootElement.appendChild(labels);
            
            // Set features map
            HashMap<String, Element> feats = new HashMap<>();
            
            for(String l_name : this.labelNames){
                
                Element label = doc.createElement("label");
                label.setAttribute("name", l_name);
                labels.appendChild(label);
                
                Element values = doc.createElement("values");
                label.appendChild(values);
                Double[] val = this.label_proba.get(l_name);
                
                // Create label = 0 field
                Element l0 = doc.createElement("value");
                l0.setAttribute("value", String.valueOf(0));
                
                Element p_l0 = doc.createElement("probability");
                p_l0.appendChild(doc.createTextNode(String.valueOf(val[0])));
                l0.appendChild(p_l0);
                
                values.appendChild(l0);
                
                // Create label = 0 field
                Element l1 = doc.createElement("value");
                l1.setAttribute("value", String.valueOf(1));
                
                Element p_l1 = doc.createElement("probability");
                p_l1.appendChild(doc.createTextNode(String.valueOf(val[1])));
                l1.appendChild(p_l1);
                
                values.appendChild(l1);
                
                // Get other probabilities
                Element features0 = doc.createElement("features");
                l0.appendChild(features0);
                Element features1 = doc.createElement("features");
                l1.appendChild(features1);
                for(Entry<String, double[]> e : this.histograms.entrySet()){
                    
                    // Get feature's name
                    String f_name = e.getKey();
                    
                    // Create elements
                    Element f_l0 = doc.createElement("feature");
                    f_l0.setAttribute("name", f_name);
                    Element probas_f_l0 = doc.createElement("probabilities");
                    f_l0.appendChild(probas_f_l0);
                    features0.appendChild(f_l0);
                    
                    Element f_l1 = doc.createElement("feature");
                    f_l1.setAttribute("name", f_name);
                    Element probas_f_l1 = doc.createElement("probabilities");
                    f_l1.appendChild(probas_f_l1);
                    features1.appendChild(f_l1);
                    
                    // If it is the first time we met this feature
                    if(!feats.containsKey(f_name)){
                        
                        // Create feature node
                        Element feature = doc.createElement("feature");
                        feature.setAttribute("name", f_name);
                        feature.setAttribute("type", "numerical");
                        features.appendChild(feature);

                        double[] e_value = e.getValue();

                        // Get feature min
                        Element d_min = doc.createElement("min");
                        double min = e_value[0];
                        d_min.appendChild(doc.createTextNode(String.valueOf(min)));
                        feature.appendChild(d_min);

                        // Get feature max
                        Element d_max = doc.createElement("max");
                        double max = e_value[1];
                        d_max.appendChild(doc.createTextNode(String.valueOf(max)));
                        feature.appendChild(d_max);

                        // Get h
                        Element d_h = doc.createElement("h");
                        double h = e_value[2];
                        d_h.appendChild(doc.createTextNode(String.valueOf(h)));
                        feature.appendChild(d_h);

                        // Get subsets
                        Element subsets = doc.createElement("subsets");
                        feature.appendChild(subsets);
                        for(int i = 0 ; i < this.n ; i++){
                            // compute subset properties
                            double s_min = (min + i*h);
                            double s_max = (min + (i+1)*h);
                            String s_name = "["+s_min+", "+s_max+"[";

                            // create element
                            Element subset = doc.createElement("subset");
                            subset.setAttribute("name", s_name);

                            Element d_smin = doc.createElement("min");
                            d_smin.appendChild(doc.createTextNode(String.valueOf(s_min)));
                            subset.appendChild(d_smin);

                            Element d_smax = doc.createElement("max");
                            d_smax.appendChild(doc.createTextNode(String.valueOf(s_max)));
                            subset.appendChild(d_smax);

                            subsets.appendChild(subset);
                            
                            // Get proba
                            String p_key = f_name + " in " + s_name + " and " + l_name;
                            Double[] p_values = {0.0,0.0};
                            if(this.termNlabel_proba.containsKey(p_key))
                                p_values = this.termNlabel_proba.get(p_key);

                            Element p0 = doc.createElement("probability");
                            p0.setAttribute("value", s_name);
                            p0.appendChild(doc.createTextNode(String.valueOf(p_values[0])));
                            probas_f_l0.appendChild(p0);

                            Element p1 = doc.createElement("probability");
                            p1.setAttribute("value", s_name);
                            p1.appendChild(doc.createTextNode(String.valueOf(p_values[1])));
                            probas_f_l1.appendChild(p1);
                        }
                        
                        // Add element to features map
                        feats.put(f_name, feature);
                    }
                    else{
                        // Get subsets of feature
                        Element feature = feats.get(f_name);
                        NodeList subsets = feature.getElementsByTagName("subset");
                        
                        for(int i = 0 ; i < subsets.getLength() ; i++){
                            Element subset = (Element) subsets.item(i);
                            String s_name = subset.getAttribute("name");
                            
                            // Get proba
                            String p_key = f_name + " in " + s_name + " and " + l_name;
                            Double[] p_values = {0.0,0.0};
                            if(this.termNlabel_proba.containsKey(p_key))
                                p_values = this.termNlabel_proba.get(p_key);
                            
                            Element p0 = doc.createElement("probability");
                            p0.setAttribute("value", s_name);
                            p0.appendChild(doc.createTextNode(String.valueOf(p_values[0])));
                            probas_f_l0.appendChild(p0);
                            
                            Element p1 = doc.createElement("probability");
                            p1.setAttribute("value", s_name);
                            p1.appendChild(doc.createTextNode(String.valueOf(p_values[1])));
                            probas_f_l1.appendChild(p1);
                        }
                    }
                }
                
                // Do the same for nominal features
                for(Entry<String, List<String>> e : nominal_values.entrySet()){
                    // Get feature's name
                    String f_name = e.getKey();
                    
                    // Create elements
                    Element f_l0 = doc.createElement("feature");
                    f_l0.setAttribute("name", f_name);
                    Element probas_f_l0 = doc.createElement("probabilities");
                    f_l0.appendChild(probas_f_l0);
                    features0.appendChild(f_l0);
                    
                    Element f_l1 = doc.createElement("feature");
                    f_l1.setAttribute("name", f_name);
                    Element probas_f_l1 = doc.createElement("probabilities");
                    f_l1.appendChild(probas_f_l1);
                    features1.appendChild(f_l1);
                    
                    // If it is the first time we met this feature
                    if(!feats.containsKey(f_name)){
                        
                        // Create feature node
                        Element feature = doc.createElement("feature");
                        feature.setAttribute("name", f_name);
                        feature.setAttribute("type", "nominal");
                        features.appendChild(feature);

                        List<String> e_values = e.getValue();

                        // Get subsets
                        Element f_values = doc.createElement("values");
                        feature.appendChild(f_values);
                        for(String s_name : e_values){

                            // create element
                            Element f_value = doc.createElement("value");
                            f_value.appendChild(doc.createTextNode(s_name));
                            f_values.appendChild(f_value);
                            
                            // Get proba
                            String p_key = f_name + "=" + s_name + " and " + l_name;
                            Double[] p_values = {0.0,0.0};
                            if(this.termNlabel_proba.containsKey(p_key))
                                p_values = this.termNlabel_proba.get(p_key);

                            Element p0 = doc.createElement("probability");
                            p0.setAttribute("value", s_name);
                            p0.appendChild(doc.createTextNode(String.valueOf(p_values[0])));
                            probas_f_l0.appendChild(p0);

                            Element p1 = doc.createElement("probability");
                            p1.setAttribute("value", s_name);
                            p1.appendChild(doc.createTextNode(String.valueOf(p_values[1])));
                            probas_f_l1.appendChild(p1);
                        }
                        
                        // Add element to features map
                        feats.put(f_name, feature);
                    }
                    else{
                        // Get subsets of feature
                        Element feature = feats.get(f_name);
                        NodeList subsets = feature.getElementsByTagName("value");
                        
                        for(int i = 0 ; i < subsets.getLength() ; i++){
                            Element subset = (Element) subsets.item(i);
                            String s_name = subset.getTextContent();
                            
                            // Get proba
                            String p_key = f_name + "=" + s_name + " and " + l_name;
                            Double[] p_values = {0.0,0.0};
                            if(this.termNlabel_proba.containsKey(p_key))
                                p_values = this.termNlabel_proba.get(p_key);
                            
                            Element p0 = doc.createElement("probability");
                            p0.setAttribute("value", s_name);
                            p0.appendChild(doc.createTextNode(String.valueOf(p_values[0])));
                            probas_f_l0.appendChild(p0);
                            
                            Element p1 = doc.createElement("probability");
                            p1.setAttribute("value", s_name);
                            p1.appendChild(doc.createTextNode(String.valueOf(p_values[1])));
                            probas_f_l1.appendChild(p1);
                        }
                    }
                }
            }
            
            
            // convert into string
            DOMSource domSource = new DOMSource(doc);
            Transformer transformer = TransformerFactory.newInstance().newTransformer();
            transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "no");
            transformer.setOutputProperty(OutputKeys.METHOD, "xml");
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
            transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
            
            File file = new File(file_path);
            StreamResult sr = new StreamResult(file);
            
            transformer.transform(domSource, sr);
            
        } catch(Exception e) {
            LOG.error(e + " - StackTrace : " + Arrays.toString(e.getStackTrace()));
        }
    }
}
