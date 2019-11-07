/*
 * This file is part of "Transparent Algorithms Performances".
 *
 * "Transparent Algorithms Performances" is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * "Transparent Algorithms Performances" is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with "Transparent Algorithms Performances".  If not, see <https://www.gnu.org/licenses/>.
 */

package com.lamsade;

//import com.lamsade.learners.FuzzyBayes;
//import com.lamsade.learners.HistBayes;
import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Utils;

/**
 * App to make cross validation of machine learning algorithms.
 *
 * @author A. Richard
 */
public class App 
{   
    private static final Logger LOG = LogManager.getLogger(App.class);
    
    /**
     * @param args the command line arguments
     */
    public static void main( String[] args )
    {
      	try{
            // Get potential options
            int numFolds = 10; // default value
            try{
                String folds = Utils.getOption("folds", args);
                if(folds !=null && !folds.isEmpty())
                    numFolds = Integer.parseInt(folds);
            }
            catch(Exception e){
                LOG.error(e);
            }
             
	    // Get result file name
            String outFilename = "../results/crossvalidation.csv"; // default value
            try{
                String tmp = Utils.getOption("output", args);
                if(tmp != null && !tmp.isEmpty())
                        outFilename = tmp;
            }
            catch(Exception e){
                LOG.error(e);
            }


            // Set datasets to use
            HashMap<String, MultiLabelInstances> datasets = new HashMap<>();
	    datasets.put("emotions", new MultiLabelInstances("../data/emotions.arff", 6)); // 72 numerical attributes and 6 labels
	    
	    /*
	    String[][] dataset_names = {
	      {"emotions", 6},// 72 numerical attributes and 6 labels
                //"CAL500", // 68 numerical attributes and 174 labels
                //"scene", // 294 numerical attributes and 6 labels
                //"genbase", // 1186 nominal attributes and 27 labels
                //"bibtex", // 1836 nominal attributes and 159 labels
                //"birds", // 2 nominal and 258 numerical attributes and 19 labels
                //"medical", // 1449 nominal attributes and 45 labels
                //"flags", // 9 nominal and 10 numeric attributes and 7 labels
               // "consultations" // 2 nominal and 2 numeric attributes and 15 labels
            };
            
            for(String[] dataset_name : dataset_names){
                datasets.put(
                    dataset_name[0],
                    new MultiLabelInstances(
                            "../data/"+dataset_name[0]+".arff", 
                            dataset_name[1]
                        )
                );
            }
	    */
	    
				
					
            // Setup output
            String[] columns = {
                    "Hamming Loss", 
                    "Micro-averaged Precision",
                    "Micro-averaged Recall",
                    "Micro-averaged F-Measure",
                    "Macro-averaged Precision",
                    "Macro-averaged Recall",
                    "Macro-averaged F-Measure"
                };
			
			
	    		
            // Init ouput file
            File f = new File(outFilename);
            if(!f.exists())
                f.getParentFile().mkdirs();
            else
                f.delete();
            f.createNewFile();

            // Write Headers
            String output = "Learner;Dataset";
            for(int i = 0 ; i < columns.length ; i++)
                output += ";" + columns[i].replace(" ", "_")
                        + ";" + columns[i].replace(" ", "_") + "_std"
                        ;
            output += "\n";

	    PrintWriter writer = new PrintWriter(f.getPath(), "UTF-8");
            writer.print(output);
            writer.flush();
            output = "";


            // For each dataset
            for(String datasetName : datasets.keySet()){

                // Get dataset
                MultiLabelInstances dataset = datasets.get(datasetName);

                LOG.info("load dataset: {"
                        + "name: " + datasetName + ", "
                        + "nb instances: " + dataset.getNumInstances() + ", "
                        + "nb attributes: " + dataset.getFeatureAttributes().size() + ", "
                        + "nb labels: " + dataset.getNumLabels() + ", "
                        + "cardinality: " + dataset.getCardinality()
                        + "}"
                    );

                // Init Learners
                HashMap<String, MultiLabelLearnerBase> learners = new HashMap<>();

                //learners.put("FuzzyBayes", new FuzzyBayes());
                //learners.put("HistBayes", new HistBayes());
                //learners.put("MLkNN", new MLkNN()); // k-Nearest Neighboors
                //learners.put("BPMLL", new BPMLL());
                
                learners.put("RAkEL+C4.5", new RAkEL(new LabelPowerset(new J48()))); // Decision Tree
                //learners.put("RAkEL+LMT", new RAkEL(new LabelPowerset(new LMT()))); // Decision Tree
                learners.put("RAkEL+Ripper", new RAkEL(new LabelPowerset(new JRip()))); // Rules
                //learners.put("RAkEL+PART", new RAkEL(new LabelPowerset(new PART()))); // DT + Rules
                learners.put("RAkEL+NaiveBayes", new RAkEL(new LabelPowerset(new NaiveBayes()))); // Bayes Net
                //learners.put("RAkEL+KStar", new RAkEL(new LabelPowerset(new KStar()))); // instance based
                //learners.put("RAkEL+SMO", new RAkEL(new LabelPowerset(new SMO()))); // SVM
                


                // For each selected learner
                for(String learnerName : learners.keySet()){

                    // Get learner
                    MultiLabelLearnerBase learner = learners.get(learnerName);

                    LOG.info("load leaner : {" 
                        + "name : " + learnerName + ", "
                        + "tech_info : \"" + learner.getTechnicalInformation().toString() + "\""
                        + "}"
                        ); 

                    // Evaluate Learner
                    Evaluator eval = new Evaluator();
                    MultipleEvaluation results;

                    LOG.info("launch " + learnerName + " evaluation with " + datasetName + " dataset");
                    try{
                    results = eval.crossValidate(learner, dataset, numFolds);


                    // Add learner name and dataset name to output
                    output += learnerName + ";" + datasetName;

                    // Add learner results to output
                    for(int i = 0 ; i < columns.length ; i++)
                            output += ";" + results.getMean(columns[i]) 
                                    + ";" + results.getStd(columns[i]);
                    output += "\n";


                    // Write into output file
                    writer.print(output);
                    writer.flush();


                    // Clean output
                    output = "";
                    }catch(IllegalArgumentException e){
                        LOG.error(e + " - StackTrace: " + Arrays.toString(e.getStackTrace()));
                    }
                }
            }
            
            writer.close();
        }
        catch(Exception e){
            LOG.error(e + " - StackTrace: " + Arrays.toString(e.getStackTrace()));
        }
    }
}
