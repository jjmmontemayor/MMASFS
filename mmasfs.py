#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:39:32 2018

@author: jennifer
"""

import numpy
import pandas
import random
from sklearn import svm
from sklearn.metrics import classification_report
from ast import literal_eval

class MMASFS :

    data_instances          = []
    data_classes            = []
    data_classes_instances  = {}
    data_class              = 0
   
    def __init__(self, iterations, alpha, beta, phi, max_pheromone, min_pheromone, q) :

        self.loadData()

        self.ITERATIONS     = iterations
        self.ALPHA          = alpha
        self.BETA           = beta
        self.PHI            = phi
        self.MAX_PHEROMONE  = max_pheromone
        self.MIN_PHEROMONE  = min_pheromone
        self.Q              = q
        self.ANT_COUNT      = self.FEATURE_COUNT

        #setup numpy arrays
        self.data_instances = numpy.array(self.data_instances)
        self.data_classes = numpy.array(self.data_classes, dtype=int)
        self.ant_steps = numpy.zeros((self.FEATURE_COUNT), dtype=object)
        self.pheromone_matrix = numpy.zeros((self.FEATURE_COUNT,self.FEATURE_COUNT), dtype=object)
        self.heuristic_matrix = numpy.zeros((self.FEATURE_COUNT,self.FEATURE_COUNT), dtype=object)
        
        #setup solution
        self.best_solution = { 0: [], 1: 0, 2: self.FEATURE_COUNT, 3: [], 4: [] }


    def loadData(self):
        #Load and process data
        f = pandas.read_csv('datasets/wine.data.txt', sep=',', header=None)
        data = f.values
        self.data_classes = data[:,0] #all first column
        self.data_instances = data[:,1:] #the rest of the columns
        
        for c in numpy.unique(self.data_classes):
            class_instance = data[numpy.where(data[:,0] == c)]
            self.data_classes_instances[c] = class_instance[:,1:]         
        

        self.FEATURE_COUNT = self.data_instances.shape[1]

    def initAnts(self):
        feature_set = random.sample(range(self.FEATURE_COUNT), self.FEATURE_COUNT)

        for row in range(self.ant_steps.shape[0]):
                step = numpy.array([feature_set[row]])
                state = numpy.array([numpy.random.randint(2)])
                self.ant_steps[row] = {0 : step, 1: state}
                
    def initPheromoneMatrix(self) :    
        for row in range(self.pheromone_matrix.shape[0]): 
            for column in range(self.pheromone_matrix.shape[1]):
                self.pheromone_matrix[row,column] = numpy.array([[self.MIN_PHEROMONE,self.MIN_PHEROMONE],[self.MIN_PHEROMONE,self.MIN_PHEROMONE]])


    def initHeuristicMatrix(self):
        for row in range(self.heuristic_matrix.shape[0]):
            for column in range(self.heuristic_matrix.shape[1]):
                if row != column :
                    self.heuristic_matrix[row,column] = numpy.array([ [self.getHeuristicValue(0,row,0,column),self.getHeuristicValue(0,row,1,column)],
                                                                  [self.getHeuristicValue(1,row,0,column),self.getHeuristicValue(1,row,1,column)] ])
    
    def evaporatePheromone(self,current_deposit):
    
        #keep within min and max values
        deposit = current_deposit - self.PHI
        if deposit <= self.MIN_PHEROMONE : deposit = self.MIN_PHEROMONE
        elif deposit >= self.MAX_PHEROMONE: deposit = self.MAX_PHEROMONE
             
        return deposit;  

    def getHeuristicValue(self,state_from, node_from, state_to, node_to) :

        total_f_score = 0

        for f in range(0, self.FEATURE_COUNT):
            total_f_score += self.getFScore(f)

        if (state_from == 0 and state_to == 0) or (state_from == 1 and state_to == 0) :
            return (state_from / self.FEATURE_COUNT) * total_f_score
        elif (state_from == 0 and state_to == 1) or (state_from == 1 and state_to == 1) :
            return self.getFScore(node_to)

    def getFScore(self, feature) :

        mean_all_classes = numpy.mean(self.data_instances, axis = 0)
        n = d = nps = 0

        for k in self.data_classes_instances.keys():
            mean_per_class = numpy.mean(self.data_classes_instances[k], axis = 0)
            n += numpy.power((mean_per_class[feature] -  mean_all_classes[feature]),2)

            #data has complete features per instance
            feature_instance_in_k = len(self.data_classes_instances[k])

            for l in self.data_classes_instances[k] :
                nps += numpy.power((l[feature] - mean_per_class[feature]), 2)
            
            d += (1 / (feature_instance_in_k - 1 ) ) * nps

        
        return n/d
    
    def getSolutionQuality(self, classes, new_instances) :
        clf = svm.SVC()
        clf.fit(new_instances,classes)
        return clf.score(new_instances,classes)
    
    def getIterationBestSolution(self) :
        score_vector = []
        reduction_vector = []
        current_best_solution = { 0: [], 1: 0, 2: 12, 3: [], 4: [] }
    
        #get selected features    
        for a in range(0, self.ANT_COUNT) :
            new_features        = numpy.nonzero(self.ant_steps[a][1])
            new_data_instances  = self.data_instances[:, new_features[0]]
            score_vector.append(self.getSolutionQuality(self.data_classes, new_data_instances))
            reduction_vector.append((self.FEATURE_COUNT - len(new_features[0])) / self.FEATURE_COUNT)
    
        ants_max_score_indices = numpy.where(score_vector == numpy.max(score_vector))
        ants_max_reduction = numpy.take(reduction_vector, ants_max_score_indices[0])
        ants_max_reduction_indices = numpy.where(ants_max_reduction == numpy.max(ants_max_reduction))
    
        for i in numpy.take(ants_max_score_indices[0],ants_max_reduction_indices[0]):
    
            features_selected = numpy.take(self.ant_steps[i][0],numpy.nonzero(self.ant_steps[i][1]))
            features_selected_count = len(features_selected[0])
            features_selected_score = score_vector[i]
                       
            if(not numpy.array_equal(current_best_solution[0],features_selected)) :
                if(features_selected_count <= current_best_solution[2] and 
                   features_selected_score >= current_best_solution[1]) :
                    current_best_solution[0] = features_selected
                    current_best_solution[1] = features_selected_score
                    current_best_solution[2] = features_selected_count
                    #path                     
                    current_best_solution[3] = self.ant_steps[i][0]
                    #state
                    current_best_solution[4] = self.ant_steps[i][1]
            
        return current_best_solution

    def updatePheromoneMatrix(self, solution) :
    
        #evaporate pheromones
        for row in range(self.pheromone_matrix.shape[0]): 
            for column in range(self.pheromone_matrix.shape[1]):
                self.pheromone_matrix[row,column][0,0] = self.evaporatePheromone(self.pheromone_matrix[row,column][0,0])
                self.pheromone_matrix[row,column][0,1] = self.evaporatePheromone(self.pheromone_matrix[row,column][0,1])
                self.pheromone_matrix[row,column][1,0] = self.evaporatePheromone(self.pheromone_matrix[row,column][1,0])
                self.pheromone_matrix[row,column][1,1] = self.evaporatePheromone(self.pheromone_matrix[row,column][1,1])

        for f in range(0,self.FEATURE_COUNT-1):
            current_feature_state  = solution[4][f]
            next_feature_state     = solution[4][f+1]
            current_feature        = solution[3][f]
            next_feature           = solution[3][f+1]

        if next_feature_state == 1 :
            current_deposit = self.pheromone_matrix[current_feature,next_feature][current_feature_state,next_feature_state]
            deposit = (1 - self.PHI) * current_deposit + self.Q / solution[1]
            self.pheromone_matrix[current_feature,next_feature][current_feature_state,next_feature_state] = deposit
              
                            
    def getNextStep(self, prev_steps, prev_steps_states, next_steps):
        current_position = prev_steps[-1]
        current_position_state = prev_steps_states[-1]
        next_steps_scores_0 = next_steps_scores_1 = dict.fromkeys(next_steps, 0)

        for next_step in next_steps:
            next_steps_scores_0[next_step] = self.getNextStepProbability(current_position, current_position_state, next_step, 0, next_steps)
            next_steps_scores_1[next_step] = self.getNextStepProbability(current_position, current_position_state, next_step, 1, next_steps)
        
        next_steps_scores_0_max = max(next_steps_scores_0,key=next_steps_scores_0.get)
        next_steps_scores_1_max = max(next_steps_scores_1,key=next_steps_scores_1.get)

        if next_steps_scores_0[next_steps_scores_0_max] == next_steps_scores_1[next_steps_scores_1_max]:
            step = next_steps_scores_0_max, numpy.random.randint(2)
        elif next_steps_scores_0[next_steps_scores_0_max] > next_steps_scores_1[next_steps_scores_1_max]:
            step = next_steps_scores_0_max, 0
        else:
            step = next_steps_scores_1_max, 1
        
        #print('selected step/state ', step, 'prev steps', prev_steps, 'next steps', next_steps)
        return step

    def getNextStepProbabilityPheromoneHeuristicWeights(self, current_step, next_step, current_step_state, next_step_state):
        return numpy.power(self.pheromone_matrix[current_step,next_step][current_step_state, next_step_state], self.ALPHA) * numpy.power(self.heuristic_matrix[current_step,next_step][current_step_state, next_step_state], self.BETA)

    def getNextStepProbability(self, current_step, current_step_state, next_step, next_step_state, possible_steps):

        n = self.getNextStepProbabilityPheromoneHeuristicWeights(current_step, next_step, current_step_state, next_step_state)
        step_0 = step_1 = 0 #woooooo careful!

        for step in possible_steps:
            step_0 += self.getNextStepProbabilityPheromoneHeuristicWeights(current_step, step, current_step_state, 0)
            step_1 += self.getNextStepProbabilityPheromoneHeuristicWeights(current_step, step, current_step_state, 1)

        return n / (step_0 + step_1)
    
    def evaluateSolution(self):
        eval_data_instances  = self.data_instances[:, self.best_solution[0][0]]
        
        clf = svm.SVC()
        clf.fit(eval_data_instances,self.data_classes)
                
        f = pandas.read_csv('datasets/wine.text.txt', sep=',', header=None)
        data = f.values
        test_data_classes = data[:,0] #all first column
        test_data_instances = data[:,1:] #the rest of the columns
        test_data_pred_classes = clf.predict(test_data_instances[:, self.best_solution[0][0]])
        target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(test_data_classes, test_data_pred_classes, target_names=target_names))

#        

    def run(self):
                
        for i in range(0,self.ITERATIONS):

            self.initAnts()
            self.initPheromoneMatrix()
            self.initHeuristicMatrix()
            
            for step in range(0, self.ant_steps.shape[0]):
                         
                while True:
                    next_steps = numpy.setxor1d(self.ant_steps[step][0],range(self.FEATURE_COUNT))
    
                    if not len(next_steps) : break
                    else :
                       next_step, next_state = self.getNextStep(self.ant_steps[step][0], self.ant_steps[step][1], next_steps)
                       self.ant_steps[step][0] = numpy.append(self.ant_steps[step][0],[next_step])
                       self.ant_steps[step][1] = numpy.append(self.ant_steps[step][1],[next_state])
            
            local_best = self.getIterationBestSolution()
            if not numpy.array_equal(self.best_solution[0], local_best[0]) :
                if local_best[1] > self.best_solution[1] and local_best[2] <= self.best_solution[2] :
                    self.best_solution = local_best
                self.updatePheromoneMatrix(local_best)    
                    
            print('local best solution', local_best)
            print('best solution', self.best_solution)
            print('------------------ Iteration ', i, '------------------')
            
        self.evaluateSolution()


a = MMASFS(50, 1, .5, .049, 6, .1, .8)
a.run()