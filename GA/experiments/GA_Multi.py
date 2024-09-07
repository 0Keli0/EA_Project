
from Individual import *
import concurrent.futures
import multiprocessing

class GA:
    def __init__(self, path,block_w_size=3,block_l_size=3):
        if(isinstance(path,str)):   
            target = Image.open(path).convert('RGBA') 
            lenght,width = target.size
            target = target.resize((lenght//2,width//2)) # if we wanna resize 
        else:target=path

        self.img = target
        self.target = np.array(target)
        self.length,self.width = target.size  #(l,w) // as array --> target.shape == (w,l,c)
        #for the splits and concatenation of the problem
        self.block_w_size = block_w_size
        self.block_l_size = block_l_size

    def applyRunGA(self,args):
        string,blocks =args
        self.runGA_Multi(pop=blocks,string_block=string)
    def runGA_Multiprocessing(self):
        blocks_8 = self.div_image_blocks((2,4))
        print("*** "*5 + "blocks_8"+" *** "*5)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            arg = ("b8",blocks_8)
            result_b8 = executor.map(self.applyRunGA, arg)
        print("*** "*5 + "blocks_8_END"+" *** "*5)

        population_f8 = self.merge_population(result_b8)
        print("/// "*5 + "Population1"+" /// "*5)
        population1 = self.runGA_Multi(population_f8,string_block="_b8")
        print("/// "*5 + "Population1_END"+" /// "*5)
        candidate = population1[0]
        candidate.image.save("pictures/multi/fitness_"+ "Block8_EndSolution"+".png")

        blocks_4 = self.div_image_blocks((2,2))
        print("*** "*5 + "blocks_4"+" *** "*5)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result_b4 = executor.map(lambda t: t.runGA_Multi(pop=population1,string_block="b4_b8"), blocks_4)
        print("*** "*5 + "blocks_4_END"+" *** "*5)

        population_f4 = self.merge_population(result_b4)
        print("/// "*5 + "Population2"+" /// "*5)
        population2 = self.runGA_Multi(population_f4,string_block="_b4_b8")
        print("/// "*5 + "Population2_END"+" /// "*5)
        candidate = population2[0] 
        candidate.image.save("pictures/multi/fitness_"+str(candidate.fitness)+ "_EndSolution"+".png")
        return population2
    def runGA_Multi(self, pop=[] ,p_size = 100, generations=15000,CO_B = 0.6, CO_P = 0.30, CO_PW = 0.05, M_R=0.05,stop_exploration = 0,stop_random_Hyper = 0, tournament_size = 6, horizontal_prob=0.5,string_block=""):
        population = pop; size = len(population)
        #in the case the p_size that have to be is more than the pop given we extend the population with more new random values --> More exploration
        if(size < p_size): 
            population.extend(self.initialization(p_size-size))
            #to mantain the best candidate at the begining we compare the best from the initialization and the best of the given population 
            if(population[p_size-size-1] < population[0]):
                population.insert(0,population.pop(p_size-size-1))
        #the list that we will return with the improvement of the generations
        best_population = [population[0] ]
        elitism = False #not elitsm aproach
        #initialize CO_Probs
        CO_B_Act = CO_B ;CO_P_Act = CO_P+CO_B_Act; CO_PW_Act=CO_P_Act+CO_P_Act+CO_PW  
        i = 0     
        while i < generations and len(best_population)<p_size:
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
                if(i < stop_random_Hyper):  
                    CO_B_Act = random.uniform(0,1)
                    CO_P_Act = random.uniform(CO_B_Act,1)
                    CO_PW_Act = random.uniform(CO_P_Act,1)
                CO = random.uniform(0,1) #how to make sure that at least have one type of crossover
                if(CO<CO_B_Act):#CrossOver_Blend
                    child = self.crossover_Blend(parent1,parent2,elitism)
                    while(child==None):
                        p1=self.tournament_selection(population,tournament_size)
                        p2=self.tournament_selection(population,tournament_size)
                        child = self.crossover_Blend(p1,p2,elitism)
                elif(CO<CO_P_Act):#CrossOver Point 
                    child = self.crossover_Point(parent1,parent2,horizontal_prob,elitism)
                    while(child==None):
                        p1=self.tournament_selection(population,tournament_size)
                        p2=self.tournament_selection(population,tournament_size)
                        child = self.crossover_Point(p1,p2,horizontal_prob,elitism)

                elif(CO<=CO_PW_Act): #CrossOver PixelWise 
                    child = self.crossover_pixel_wise(parent1,parent2,elitism)
                    while(child==None):
                        p1=self.tournament_selection(population,tournament_size)
                        p2=self.tournament_selection(population,tournament_size)
                        child = self.crossover_pixel_wise(p1,p2,elitism)
                else:#fight to survive 
                    child = parent1 if parent1.fitness < parent2.fitness else parent2
                    
                #Mutation
                random_m = random.uniform(0,1)
                if(random_m<=M_R):
                    mutationType = random.randint(0,1)
                    if mutationType < 1:#Mutation Pixel
                        child = self.mutation_pixels(child)  
                    else:#Mutation Poligons
                        child = self.mutation_poligon(child)  
                if(len(new_population)==0 or new_population[0].fitness>child.fitness):
                    new_population.insert(0,child)
                else: new_population.append(child)

            if(new_population[0] < best_population[0]):best_population.insert(0,new_population[0])

            population = new_population

            if((i%1000==0 or i == generations -1) and string_block[0]=="_"):
                candidate=best_population[0]
                candidate.image.save("pictures/multi/"+string_block+"/"+"fitness_"+str(candidate.fitness)+ "_epouch_"+str(i) +".png")
        return best_population   
    def initialization(self,pop_size, HoF = False):
        population = []
        
        for _ in range(pop_size):
            new_Ind = Individual(self.length,self.width)
            new_Ind.get_fitness(self.target)

            if (HoF == True and len(population)!=0): 
                if(population[0].fitness >= new_Ind.fitness):
                    population.insert(0,new_Ind)
                else: population.append(new_Ind) #Ya probaremos con insert, pero creo que lo hace más lento

            else: 
                population.append(new_Ind) 

        return population

    @staticmethod
    def tournament_selection(population, tournament_size=6):
        ind = population[-1]
        winner = Individual(ind.length,ind.width,img=ind.image) 
        winner.fitness = None

        for _ in range(tournament_size):
            cand = random.choice(population)
            
            if(winner.image == None or cand.fitness < winner.fitness):
                winner = cand
        return winner
    
    def crossover_Blend(self,ind1,ind2,elitism=True):
        child_image = Image.blend(ind1.image,ind2.image,random.random())
    
        child = Individual(self.length,self.width,img=child_image)
        child.get_fitness(self.target)
        #print(child.fitness)
        #elitism
        if(child.fitness == min(child.fitness,ind1.fitness,ind2.fitness) and elitism):
            return child
        return None


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

            split_point = random.randint(1,self.length)

            ones,zeros = np.ones((self.width,split_point)), np.zeros((self.width,self.length-split_point))
            first = np.hstack((ones,zeros))
          
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
        if (child.fitness == min(child.fitness,ind1.fitness,ind2.fitness) and elitism):
            return child
        
        return None
    
    def crossover_pixel_wise(self,ind1,ind2,elitism = True):

        first = np.random.randint(2,size = (self.target.shape))
        second = 1-first
        half_child1, half_child2 = np.multiply(first,ind1.array),np.multiply(second,ind2.array)

        offspring = np.add(half_child1,half_child2)
        child = Individual(self.length,self.width,img=offspring)
        child.get_fitness(self.target)
        #elitism
        if (child.fitness == min(child.fitness,ind1.fitness,ind2.fitness) and elitism):
            return child
        return None
    
    def mutation_poligon(self,ind,n_poligons=6):
        img =  ind.image.copy() #to not really alterate the individual and drop a new one
        poligon_num = random.randint(1,n_poligons)
        region = random.randint(1,(self.width+self.length)//5)

        for i in range(poligon_num):
            num_points = random.randint(3,6)
            x,y = random.randint(0,self.length) , random.randint(0,self.width)

            xy = [(random.randint(x-region,x+region),random.randint(y-region,y+region))
                    for _ in range(num_points)]
            
            #adding more poligons to the image img
            img1 = ImageDraw.Draw(img)
            img1.polygon( xy,fill=ind.random_string_color())

        child = Individual(self.length,self.width,img=img)
        child.get_fitness(self.target)

        return child
    
    def mutation_pixels(self,ind, pixels=40,change=20):
        indArray = ind.array.copy() 
        for _ in range(pixels):
            y= random.randint(0,self.length-1)
            x=random.randint(0,self.width-1)
            z= random.randint(0,self.target.shape[-1]-1)
            indArray[x][y][z] = indArray[x][y][z] + random.randint(-change,change)
        #cast values are in the proper limit for a image
        imgArray = np.clip(indArray,0,255) #not cross the possible range of values that a pixel can have

        child = Individual(self.length,self.width,img=imgArray)
        child.get_fitness(self.target)

        return child
    
    def div_image_blocks(self,next_block_size = (3,3)):
        w_size=self.block_w_size; l_size= self.block_l_size
        img = self.target.copy()
        blocks = []
        b_w_size, b_l_size = self.width // w_size , self.length // l_size
        for iw in range(w_size):
            for il in range(l_size):
                x = b_w_size * iw
                y = b_l_size * il
                block = img[int(x):int(x+b_w_size), int(y):int(y+b_l_size)]
                new_GA = GA(Image.fromarray(block),next_block_size[0],next_block_size[1])
                blocks.append(new_GA)
        return blocks
    def concat_blocks(self,blocks):
        w_size=self.block_w_size; l_size= self.block_l_size
        rows = []
        for li in range(l_size):
            num_block_row = li*w_size
            row_i = np.concatenate([*map(lambda x: x.array,blocks[num_block_row:num_block_row+w_size])], axis=1)
            rows.append(row_i)
        sol_concat = np.concatenate(rows,axis=0)
        candidate = Individual(self.length,self.width,img = sol_concat)
        return candidate
    def merge_population(self,HoF_population_subBlocks):
        new_HoF_population = []
        for cl_individual in zip(*HoF_population_subBlocks):
            new_HoF_population.append(self.concat_blocks(cl_individual))
        return new_HoF_population    


if __name__ == '__main__':

    GA_Sol = GA("gioconda.png",2,4)
    population2 = GA_Sol.runGA_Multiprocessing()
    population2[0].image.show()



'''
    def runGA_Multi1(self, pop=[] ,p_size = 100, generations=15000,CO_B = 0.6, CO_P = 0.35, CO_PW = 0.05, M_R=0.05,stop_exploration = 2, HoF_or_Elitism=True):
        population = pop;size = len(population) 
        if(size<p_size): 
            population.extend(self.initialization(p_size-size))
        #after the merge and the need divisions we don't know anymore if the first value is the best or not
        best_value = None; index = None
        for ind,p in enumerate(population):
            if(best_value==None or p.fitness < best_value):
                index = ind
        population.insert(0,population.pop(index)) #with this, in any case we put the best fitness at the begining of the population
#ESTO NO TE SIRVE PA NA
        best_sol = []
        if(HoF_or_Elitism):
            elitism = []
            best_sol = population[0]
            elitism.append((0,best_sol))

            for i in range(generations):
                
                new_population = [] 
            
                while len(new_population) < p_size:
                    
                    randCO = random.uniform(0,1)

                    parent1 = self.tournament_selection(population) #tournament size = 6
                    parent2 = self.tournament_selection(population)

                    if CO_B: #CrossOver_Blend
                        child = self.crossover_Blend(parent1,parent2)
                        while(child == None):
                            parent1 = self.tournament_selection(population) 
                            parent2 = self.tournament_selection(population)
                            child = self.crossover_Blend(parent1,parent2)

                    elif(randCO < CO_B+CO_P): #CrossOver Point
                        child = self.crossover_Point(parent1,parent2,0.5)
                        while(child == None):
                            parent1 = self.tournament_selection(population) 
                            parent2 = self.tournament_selection(population)
                            child = self.crossover_Point(parent1,parent2,0.5) #same probability to divided horizontaly or verticaly
                    
                    else:#CrossOver PixelWise
                        child = self.crossover_pixel_wise(parent1,parent2)
                        while(child == None):
                            parent1 = self.tournament_selection(population) 
                            parent2 = self.tournament_selection(population)
                            child = self.crossover_pixel_wise(parent1,parent2)

                    #Mutations
                    randMut = random.uniform(0,1)
                    if(randMut <= M_R):
                        mutationType = random.randint(0,1)
                        if mutationType < 1:#Mutation Pixel
                            child = self.mutation_pixels(child)  
                        else:#Mutation Poligons
                            child = self.mutation_poligon(child)  
                    print(child.fitness)     
                    if(population[0].fitness > child.fitness or len(new_population)==0): new_population.insert(0,child)
                    else: new_population.append(child)
                # is not gonna be necesary if we add always the best solution to a extreme of the new_population list best_sol = min(new_population,key=lambda x: x.fitness)
                #only in case where the new best solution is something better than the best sol introduced in the hall_of_fame
                best_sol = new_population[0]
                if elitism[-1][1].fitness >= best_sol.fitness: elitism.append((i+1,best_sol))

                population = new_population
            
                if (i%100==0 or i == generations-1):
                    print('-'*10,'epouch ',i,'-'*10)
                    candidate=elitism[-1][1]
                    candidate.image.save("pictures/fitness_"+str(candidate.fitness)+ "_epouch_"+str(i) +".png")
            best_sol = elitism
        else:
            Hall_of_Fame = []

            best_sol = Hall_of_Fame
        return best_sol


    
'''