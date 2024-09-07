
from Individual import *


class GA:
    def __init__(self, path):
        if(isinstance(path,str)):   
            target = Image.open(path).convert('RGBA') 
            lenght,width = target.size
            target = target.resize((lenght//2,width//2)) # if we wanna resize 
        else:target=path

        self.img = target
        self.target = np.array(target)
        self.length,self.width = target.size  #(l,w) // as array --> target.shape == (w,l,c)
        #for the splits and concatenation of the problem
        


    def runGA(self, population_size, generations,CO_B = 0.6, CO_P = 0.35, CO_PW = 0.05, M_R=0.05):
        population = self.initialization(population_size,HoF=True)
        hall_of_fame = []
        best_sol = population[0]
        hall_of_fame.append((0,best_sol))

        for i in range(generations):
            
            new_population = [] 
        
            while len(new_population) < population_size:
                
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
                    child = self.crossover_Pixel_Wise(parent1,parent2)
                    while(child == None):
                        parent1 = self.tournament_selection(population) 
                        parent2 = self.tournament_selection(population)
                        child = self.crossover_Pixel_Wise(parent1,parent2)

                #Mutations
                randMut = random.uniform(0,1)
                if(randMut <= M_R):
                    mutationType = random.randint(0,1)
                    if mutationType < 1:#Mutation Pixel
                        child = self.mutation_pixels(child)  
                    else:#Mutation Poligons
                        child = self.mutation_Polygon(child)  
                print(child.fitness)     
                if(population[0].fitness > child.fitness or len(new_population)==0): new_population.insert(0,child)
                else: new_population.append(child)
            # is not gonna be necesary if we add always the best solution to a extreme of the new_population list best_sol = min(new_population,key=lambda x: x.fitness)
            #only in case where the new best solution is something better than the best sol introduced in the hall_of_fame
            best_sol = new_population[0]
            if hall_of_fame[-1][1].fitness >= best_sol.fitness: hall_of_fame.append((i+1,best_sol))

            population = new_population
           
            if (i%100==0 or i == generations-1):
                print('-'*10,'epouch ',i,'-'*10)
                candidate=hall_of_fame[-1][1]
                candidate.image.save("pictures/fitness_"+str(candidate.fitness)+ "_epouch_"+str(i) +".png")
        return hall_of_fame


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
        winner = None

        for _ in range(tournament_size):
            cand = random.choice(population)
            
            if(winner == None or cand.fitness < winner.fitness):
                winner = cand
        return winner
    
    def crossover_Blend(self,ind1,ind2):
        child_image = Image.blend(ind1.image,ind2.image,random.random())
    
        child = Individual(self.length,self.width,img=child_image)
        child.get_fitness(self.target)
        #print(child.fitness)
        #elitism
        if(child.fitness == min(child.fitness,ind1.fitness,ind2.fitness)):
            return child
        return None


    def crossover_Point(self,ind1,ind2,horizontal_prob=0.5):

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
        if child.fitness == min(child.fitness,ind1.fitness,ind2.fitness):
            return child
        
        return None
    
    def crossover_Pixel_Wise(self,ind1,ind2):

        first = np.random.randint(2,size = (self.target.shape))
        second = 1-first
        half_child1, half_child2 = np.multiply(first,ind1.array),np.multiply(second,ind2.array)

        offspring = np.add(half_child1,half_child2)
        child = Individual(self.length,self.width,img=offspring)
        child.get_fitness(self.target)
        #elitism
        if child.fitness == min(child.fitness,ind1.fitness,ind2.fitness):
            return child
        return None
    
    def mutation_Polygon(self,ind,n_poligons=6):
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
    
    
 