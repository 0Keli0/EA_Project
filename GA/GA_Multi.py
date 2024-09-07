
from Individual import *
#import concurrent.futures
import multiprocessing
import pickle
    

class GA_Multi:
    """
    GA defined by: 
        img -> The target image
        length -> The length of the target image
        width -> The width of the target image
        block_w_size -> The width portion to split the target image 
        block_l_size ->  The length portion to split the target image
    """
    def __init__(self, path,block_w_size=3,block_l_size=3):
        if(isinstance(path,str)):   
            target = Image.open(path).convert('RGBA') 
            target = target.resize((972,1383)) 
        else:target=path

        self.img = target
        self.target = np.array(target,dtype=np.uint8)
        self.length,self.width = target.size  #(l,w) // as array --> target.shape == (w,l,c)
        #for the splits and concatenation of the problem
        self.block_w_size = block_w_size
        self.block_l_size = block_l_size

    
    def runGA_Multiprocessing(self):


    ########################BLOCKS 9 START########################    
        blocks_9 = self.div_image_blocks((2,2))
        print("*** "*5 + "blocks_9"+" *** "*5)
        return_dict = multiprocessing.Manager().dict()
        process = []
        result_b9 = []
        for i in range(9):
            p = multiprocessing.Process(target=blocks_9[i].runGA_Multi,args=(i,return_dict,[],70,5000,))
            p.start()
            process.append(p)
        for i,pr in enumerate(process):
            pr.join()
            result_b9.append(return_dict[i])  
            print(str(i)+"_b9")  
        '''
        t=0
        
        while(t<9):
            result_b9[t][0].image.save("chaining/Process_"+str(t)+" fitness_"+str(result_b9[t][0].fitness)+ " results_b9_EndSolution"+".png")
            t = t+1
        '''

        print("*** "*5 + "blocks_9_END"+" *** "*5)
    ########################BLOCKS 9 END########################    

        population_f9 = self.merge_population(result_b9)
        
        candidate = population_f9[0]
        candidate.get_fitness(self.target)
        candidate.image.save("pictures/multi/Colour/best_sol/_fitness_"+str(candidate.fitness)+ "Block9_EndSolution"+".png")
        #candidate.image.save("chaining/"+" fitness_"+str(candidate.fitness)+ "Merge_results_b9_EndSolution"+".png")


        
    ########################FIRST MERGE START########################    
        print("/// "*5 + "Population1"+" /// "*5)
        population1_index = self.runGA_Multi(0,{},population_f9,80,5000,string_block="_b9")
        print("/// "*5 + "Population1_END"+" /// "*5)

        population1 = population1_index[0]
        candidate = population1[0]
        candidate.get_fitness(self.target)
        candidate.image.save("pictures/multi/Colour/best_sol/_fitness_"+str(candidate.fitness)+ "_Pop1_EndSolution"+".png")
    ########################FIRST MERGE END########################    

        
        #population1[0].image.save("PRUEBA_DIV_B4.png")


    ########################BLOCKS 4 START########################    
        pop1 = []
        div_pop1 = [*map(lambda  i: i.div_in_blocks((2,2)),population1)]
        for block in zip(*div_pop1):
            pop1.append(block)
        self.block_w_size,self.block_l_size=(2,2)  #resize the blocks measures
        blocks_4 = self.div_image_blocks((2,2))

        print("*** "*5 + "blocks_4"+" *** "*5)

        return_dict = multiprocessing.Manager().dict()
        result_b4 = []
        process = []
        for i in range(4):     
            p = multiprocessing.Process(target=blocks_4[i].runGA_Multi,args=(i,return_dict,[*pop1[i]],50,5000,))
            p.start()
            process.append(p)
        for i,pr in enumerate(process):
            pr.join()    
            result_b4.append(return_dict[i]) 
            print(str(i)+"_b4")  

        
        print("*** "*5 + "blocks_4_END"+" *** "*5)
    ########################BLOCKS 4 END########################    


        population_f4 = self.merge_population(result_b4)
        candidate = population_f4[0]
        candidate.get_fitness(self.target)

        candidate.image.save("pictures/multi/Colour/best_sol/_fitness_"+str(candidate.fitness)+ "Block4_EndSolution"+".png")
        

    ########################SECOND MERGE START########################    
        print("/// "*5 + "Population2"+" /// "*5)
        population2_index = self.runGA_Multi(0,{},population_f4,30,10000,string_block="_b4_b9")
        population2 = population2_index[0]
        print("/// "*5 + "Population2_END"+" /// "*5)

        candidate = population2[0]
        candidate.get_fitness(self.target)
        candidate.image.save("pictures/multi/best/_fitness_"+str(candidate.fitness)+ "_Pop2_EndSolution"+".png")
    ########################SECOND MERGE END########################    

        bests_per_Generation_9 = population1[1]
        bests_per_Generation_4 = population2[1]
        return population2,bests_per_Generation_9,bests_per_Generation_4
        
    #This method is the Base of the evolutionary algorithm. After the definition of the population (in case where it is given,it just make sure that the "p_size" the same, if not create more),
    # we start the generations loop. In the case where we would like to have some generations without elitism in our methods, we just need to define how many in the "stop_exploration". Also, if 
    # we want to have random crossover and mutation rate we just need to input the number of generations in "stop_random_Hyper". Then, the generation of the new population starts
    # Having 3 differents types of crossovers, first we select the two parents for the sexual crossover. Then, the Crossover_ratios will determinate which type of the crossover is gonna be done.
    # In the case where there is no a crossover, then we choose the children as the best individual between the parents. if you are in an elitist approach, then every we will loop the crossover where we are
    # being stuck there till we find a better offspring than the new parents selected by the tournament. After that, the offspring have the chance to have a mutation, where there both approach are uniformly selected.
    #Finally, if the child is a better solution than first individual in the list of "best_population" it will be insert as the new first. Then, after a defined number of generations, this 
    #best solution will be appended in a list called "best_pop_per_print" with the generation, and also we will save the best solution as an image.
    #The last part only would happen if the problem is the "General Problem" an there is no multiprocesures involved, because in this case the result of the "Subproblem" will be obtained from the "return_dict"
    def runGA_Multi(self,index_procex,return_dict, pop=[] 
                    ,p_size = 100, generations=15000,CO_B = 0.6, CO_P = 0.30, CO_PW = 0.05, M_R=0.05,
                                                        stop_exploration = 0,stop_random_Hyper = 0, tournament_size = 6, horizontal_prob=0.5,string_block=" "):
        population = pop; size = len(population)
        best_pop_per_print = []
        #in the case the p_size that have to be is more than the pop given we extend the population with more new random values --> More exploration
        if(size < p_size): 
            population.extend(self.initialization(p_size-size))
            #to mantain the best candidate at the begining we compare the best from the initialization and the best of the given population 
            if(population[p_size-size-1].fitness < population[0].fitness):
                population.insert(0,population.pop(p_size-size-1))
        if(size > p_size):
            population = population[:p_size-1]
        #the list that we will return with the improvement of the generations
        best_population = [population[0] ]
        elitism = False #not elitsm aproach
        #initialize CO_Probs
        CO_B_Act = CO_B ;CO_P_Act = CO_P+CO_B_Act; CO_PW_Act=CO_P_Act+CO_P_Act+CO_PW  
        i = 0     
        while i < generations : #and len(best_population)<p_size YA VEREMOS
            if(i == stop_exploration): elitism=True # elitism aproach 

            if(i == stop_random_Hyper): #random hyperparamiters, not anymore
                CO_B_Act = CO_B
                CO_P_Act = CO_P+CO_B_Act
                CO_PW_Act=CO_P_Act+CO_P_Act+CO_PW

            new_population = []
            while len(new_population) < p_size:
                child = None
                parent1 = self.tournament_selection(population,tournament_size) #tournament size = 6
                parent2 = self.tournament_selection(population,tournament_size)
                if(i < stop_random_Hyper): # i < stop_random_Hyper
                    CO_B_Act = random.uniform(0,1)
                    CO_P_Act = random.uniform(CO_B_Act,1)
                    CO_PW_Act = random.uniform(CO_P_Act,1)
                CO = random.uniform(0,1) #how to make sure that at least have one type of crossover
                if(CO<CO_B_Act ):#CrossOver_Blend
                    child = self.crossover_Blend(parent1,parent2,elitism)
                    while(child==None):
                        parent1=self.tournament_selection(population,tournament_size)
                        parent2=self.tournament_selection(population,tournament_size)
                        child = self.crossover_Blend(parent1,parent2,elitism)
                elif(CO<CO_P_Act ):#CrossOver Point 
                    child = self.crossover_Point(parent1,parent2,horizontal_prob,elitism)
                    while(child==None):
                        parent1=self.tournament_selection(population,tournament_size)
                        parent2=self.tournament_selection(population,tournament_size)
                        child = self.crossover_Point(parent1,parent2,horizontal_prob,elitism)

                elif(CO<=CO_PW_Act): #CrossOver PixelWise 
                    child = self.crossover_Pixel_Wise(parent1,parent2,elitism)
                    while(child==None):
                        parent1=self.tournament_selection(population,tournament_size)
                        parent2=self.tournament_selection(population,tournament_size)
                        child = self.crossover_Pixel_Wise(parent1,parent2,elitism)
                else:#fight to survive 
                    child = parent1 if parent1.fitness < parent2.fitness else parent2
                    
                #Mutation
                random_m = random.uniform(0,1)
                if(random_m<=M_R):
                    mutationType = random.randint(0,1)
                    if mutationType < 1:#Mutation Pixel
                        child = self.mutation_Pixels(child)  
                    else:#Mutation Polygon
                        child = self.mutation_Polygon(child)  

                if(len(new_population)==0 or new_population[0].fitness>child.fitness):
                    new_population.insert(0,child)
                else: new_population.append(child)

            if(new_population[0].fitness < best_population[0].fitness):best_population.insert(0,new_population[0])

            population = new_population

            if((i%100==0 or i == generations -1) and string_block[0]=="_"):
                candidate=best_population[0]
                if(index_procex!=-1):
                    best_pop_per_print.append((i,candidate))
                
                candidate.image.save("pictures/multi/Colour/"+string_block+"/"+"num_PoP_"+str(p_size)+"_epouch_"+str(i)+" of "+str(generations) +"_fitness_"+str(candidate.fitness)+".png")

            i = i+1
        return_dict[index_procex] = population
        return population,best_pop_per_print
    

########INITIALIZATION METHOD##########    
    #This Method initialize a random population with a given number of members and with the same margin that the problem GA has.
    #If the parameter "best_first" is True, then the first value gave after create the population will be the best of the random 
    #population generated.
    def initialization(self,pop_size, best_first = False, polygon_n = 6): 
        population = []
        for _ in range(pop_size):
            new_Ind = Individual(self.length,self.width,polygon_n)
            new_Ind.get_fitness(self.target)
            if (best_first == True and len(population)!=0): 
                if(population[0].fitness >= new_Ind.fitness):
                    population.insert(0,new_Ind)
                else: 
                    population.append(new_Ind) 
            else: 
                population.append(new_Ind) 
        return population
    
########SELECTION METHOD##########
    @staticmethod
    #Static Method that generate a tournament selection from a population and a tournament size given. 
    def tournament_selection(population, tournament_size=6):
        ind = population[-1] #Select a Individual from the population
        #Use there properties to create a new Individual to redifine its fitness
        winner = Individual(ind.length,ind.width,img=ind.image)  
        winner.fitness = None
        for _ in range(tournament_size):
            #select a random candidate and ask if its fitness is better than the 
            # winner fitness
            cand = random.choice(population)
            if(winner.fitness == None or cand.fitness < winner.fitness):
                #exchange the competitors and give a chance as winner for a better candidate
                winner = cand
        return winner
    
########REPRODUCTION METHODS##########
    
    ####CROSSOVER METHODS START####
    #In all the CROSSOVER METHODS there is an "elitism" implementation where if the parameter "elitism=True", then  the crossover only will return the child generated by the 
    # crossover if its fitness is better than the both parents. If it's not, then would return "None"

    # CROSSOVER I: This method blend two individuals given as parameters using the Image.blend(image1,image2,alpha) function where alpha define the opacity percentage that 
    # will be selected per parents. If alpha is 0.0 the child result would be the image1 and if alpha is 1.0 the child would be image2.

    def crossover_Blend(self,ind1,ind2,elitism=True):
        child_image = Image.blend(ind1.image,ind2.image,random.random())
        child = Individual(self.length,self.width,img=child_image)
        child.get_fitness(self.target)
    
        if(child.fitness != min(child.fitness,ind1.fitness,ind2.fitness) and elitism):
            return None
        return child

    # CROSSOVER II: In this approach, first we select randomly If we are going to do the horizontal or vertical crossing by the "horizontal_prob" parameter given. 
    # Based on this, a series of columns/rows will be randomly selected to create a matrix of 1s and 0s and its opposite, with the same dimensions and channels as
    #  that of individuals. With them, we will multiply one of the parents by one and the other by the opposite, causing the columns/rows corresponding to 1s to keep 
    # those genes of that parent, while the columns/rows that are 0s will invalidate genes of that parent. In this way, and having done the same with the opposite,
    #  these can be added by maintaining the corresponding columns/rows of each one based on that random generation of matrices.

    def crossover_Point(self,ind1,ind2,horizontal_prob=0.5,elitism=True):

        #horizontal crossover point
        if random.random() <= horizontal_prob:
            #choose a column from 
            split_point = random.randint(1,self.width)
            #create a matrix of ones with the measure of the columns chosen
            ones,zeros = np.ones((split_point,self.length)),np.zeros((self.width-split_point,self.length))
            #then add to this matrix an extension of 0s columns which correspond to the rest of the columns to fit the image's size
            first = np.vstack((ones,zeros))
       
        #vertical crossover point --> same procedure but now with rows instead of columns
        else:
            #choose a row from 
            split_point = random.randint(1,self.length)
            #create a matrix of ones with the measure of the rows chosen
            ones,zeros = np.ones((self.width,split_point)), np.zeros((self.width,self.length-split_point))
             #then add to this matrix an extension of 0s rows which correspond to the rest of the rows to fit the image's size
            first = np.hstack((ones,zeros))
        #this is the way we create the opposite matrix --> where there were 0s now there are 1s and vice versa
        second = 1 - first   

        #now we add the differents channels that compound the image with the same values that we generate before
        first_part = np.dstack([first]*self.target.shape[-1])
        second_part = np.dstack([second]*self.target.shape[-1])

        #multiply the matrix with the parents to add the random part which now will correspond to the child 
        half_child1,half_child2 = np.multiply(first_part,ind1.array),np.multiply(second_part,ind2.array)
        offspring = np.add(half_child1,half_child2)
        child = Individual(self.length,self.width,img=offspring)
        child.get_fitness(self.target)

        #elitism
        if (child.fitness != min(child.fitness,ind1.fitness,ind2.fitness) and elitism):
            return None
        return child
    

    # CROSSOVER III: In this method we do something similar to the previous approach, but instead of choose a number of columns or rows from each parents, we just create a random matrix with 0s and 1s
    #  with the shape of the target and its opposite. After that, We would proceed in the same way: multiply one parent by the matrix, the other by the opposite and add them to generate the offspring

    def crossover_Pixel_Wise(self,ind1,ind2,elitism = True):
        # generate a random matrix with the target shape and defined by 0 or 1
        first = np.random.randint(2,size = (self.target.shape)) 
        second = 1-first #oppisite Matrix
        #multiply the matrix with the parents to add the random part which now will correspond to the child 
        half_child1, half_child2 = np.multiply(first,ind1.array),np.multiply(second,ind2.array)
        offspring = np.add(half_child1,half_child2) #Mix the parents
        child = Individual(self.length,self.width,img=offspring)
        child.get_fitness(self.target)
        #elitism
        if (child.fitness != min(child.fitness,ind1.fitness,ind2.fitness) and elitism):
            return None
        return child 
    ####CROSSOVER METHODS END####


    ####MUTATION METHODS START####
        #MUTATION I: This method consists of adding a random number of new more polygons between 1 and "n_polygons" to the copy of the individual "ind" given as an argument. 
        # In this case, "region" is a little different, since in this case it is chosen between 1 and one-fifth of the sum of the margin measurements. 
        # This is with the intention that the possible values ​​have a greater range, at the same time that this does not always happen since it is chosen 
        # between 1 and said value (so there will be cases in which even the allowed range will be less than the one proposed for the random generation of the individual)

    def mutation_Polygon(self,ind,n_polygons=6):
        img =  ind.image.copy() #to not really alterate the individual and drop a new one
        polygon_num = random.randint(1,n_polygons)
        region = random.randint(1,(self.width+self.length)//5)
        #SAME as the generation of random individuals
        for i in range(polygon_num):
            num_points = random.randint(3,6)
            x,y = random.randint(0,self.length) , random.randint(0,self.width)
            xy = [(random.randint(x-region,x+region),random.randint(y-region,y+region))
                    for _ in range(num_points)]
            #adding more polygons to the image img
            img1 = ImageDraw.Draw(img)
            img1.polygon( xy,fill=ind.random_string_color())

        child = Individual(self.length,self.width,img=img)
        child.get_fitness(self.target)

        return child
    
     # MUTATION II: In this last method we let the chance to change a "pixel" number of pixels
     #  chosen randomly by adding a random change between [-"change" and "change"].
     #  Just to make sure that the  values of the image not overflows, I use the function 
     # np.clip(image,lim_min,lim_max) to limit the values in the  proper bounds.

    def mutation_Pixels(self,ind, pixels=40,change=20):
        indArray = ind.array.copy() 
        for _ in range(pixels):
            y= random.randint(0,self.length-1) #random y value
            x=random.randint(0,self.width-1) #random x value
            z= random.randint(0,self.target.shape[-1]-1) #random z value
            indArray[x][y][z] = indArray[x][y][z] + random.randint(-change,change) #change the pixel
          
        #cast values are in the proper limit for a image
        imgArray = np.clip(indArray,0,255) #not cross the possible range of values that a pixel can have I thought it was necessary but it do it by itself

        child = Individual(self.length,self.width,img=imgArray)
        child.get_fitness(self.target)

        return child

########BLOCKS METHODS##########
    #With this method, using the properties of "block_w_size" and "block_l_size", 
    # it allows us to subdivide the problem into other problems by dividing the image 
    # of the problem that calls the method into (block_w_size * block_l_size) 
    # images encapsulated in objects of type GA_Multi
    def div_image_blocks(self,next_block_size = (3,3)):
        w_size=self.block_w_size; l_size= self.block_l_size
        img = self.target.copy()
        blocks = []
        #calculate the new measures of the images (notice that they have to be natural numbers)
        b_w_size, b_l_size = self.width // w_size , self.length // l_size
        for iw in range(w_size):
            for il in range(l_size):
                x = b_w_size * iw
                y = b_l_size * il
                block = img[int(x):int(x+b_w_size), int(y):int(y+b_l_size)]
                #With this portion we create a new object GA_Multi as a new problem
                new_GA = GA_Multi(Image.fromarray(block),next_block_size[0],next_block_size[1])
                #Adding all the "SubProblems" to a list 
                blocks.append(new_GA)
        return blocks
    #This method, using a list of pieces of the individuals given as an argument, 
    # chains the pieces of photos and then creates an object Individual with this image piece of the individual
    def concat_blocks(self,blocks):
        w_size=self.block_w_size; l_size= self.block_l_size
        rows = []
        for li in range(l_size):
            num_block_row = li*w_size
            row_i = np.concatenate([*map(lambda x: x.array,blocks[num_block_row:num_block_row+w_size])], axis=1)
            rows.append(row_i)
        sol_concat = np.concatenate(rows,axis=0)
        if(sol_concat.shape != self.target.shape): #In the case where there is any problem with the resizment after the merge
            sol_concat = Image.fromarray(sol_concat.astype(np.uint8)).resize((self.length,self.width))
        candidate = Individual(self.length,self.width,img = sol_concat)
        return candidate
    #This method transform a list of populations to solve SubProblems to an unique population to solve a General Problem
    def merge_population(self,population_subBlocks):
        new_HoF_population = []
        for cl_individual in zip(*population_subBlocks):
            new_HoF_population.append(self.concat_blocks(cl_individual))
        return new_HoF_population    




if __name__ == '__main__':
    GA_Sol = GA_Multi("gioconda.png",3,3)
    GA_Sol.img.show() # type: ignore
    solution = GA_Sol.runGA_Multiprocessing()

   
    

