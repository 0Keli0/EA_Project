from GA_Multi import *

GA_sol = GA_Multi("gioconda.png")
GA_Ini_Pop = GA_sol.initialization(100,True)
#GA_Ini_Pop[0].image.show()
#GA_sol.img.show()
#t = GA_Ini_Pop[0].image.save("pictures/aqui.png")

l = []
l.insert(0,1)
print(l)
print(min(1.,2,3))


GA_sol = GA_Multi("gioconda.png")
GA_Ini_Pop = GA_sol.initialization(100,True)

#Initial_Population
for i in range(5):
    GA_Ini_Pop[i].image.save("experiments/initial_population/initial_population_"+str(i)+".png")

#Individual_Split_Blocks
p1 = GA_Ini_Pop[0]
f, axarr = plt.subplots(3,3)
p1_div_3_3 = p1.div_in_blocks()
for i in range(3):
    for j in range(3):
        axarr[i,j].imshow(p1_div_3_3[3*i + j].image)

plt.savefig("experiments/initial_population/div_blocks/"+"initial_population_0"+".jpg")  


#CROSSOVERS 
    
p1 = GA_Ini_Pop[0]
p2 = GA_Ini_Pop[1]
i,j = 2,2
    #crossover_Blend
child = GA_sol.crossover_Blend(p1,p2,False)
child.get_fitness(GA_sol.target)# type: ignore
f, axarr = plt.subplots(1,4)
f.supxlabel(str(p1.fitness) +"  " +str(p2.fitness) + "  "+str(child.fitness) ) # type: ignore
f.suptitle("crossover_Blend")
axarr[0].imshow(GA_sol.target)
axarr[1].imshow(p1.image)
axarr[2].imshow(p2.image)
axarr[3].imshow(child.image) # type: ignore
plt.show()
f.savefig("experiments/reproduction/crossover/"+"crossover_Blend"+".jpg")

    #crossover_Point

child = GA_sol.crossover_Point(p1,p2, elitism=False) 

f, axarr = plt.subplots(1,4)
f.supxlabel(str(p1.fitness) +"  " +str(p2.fitness) + "  "+str(child.fitness) ) # type: ignore
f.suptitle("crossover_Point")
axarr[0].imshow(GA_sol.target)
axarr[1].imshow(p1.image)
axarr[2].imshow(p2.image)
axarr[3].imshow(child.image) # type: ignore
plt.show()
f.savefig("experiments/reproduction/crossover/"+"crossover_Point"+".jpg")


    #crossover_Pixel_Wise

child = GA_sol.crossover_Pixel_Wise(p1,p2,elitism=False)


f, axarr = plt.subplots(1,4)
f.supxlabel(str(p1.fitness) +"  " +str(p2.fitness) + "  "+str(child.fitness) ) # type: ignore
f.suptitle("crossover_pixel_wise")
axarr[0].imshow(GA_sol.target)
axarr[1].imshow(p1.image)
axarr[2].imshow(p2.image)
axarr[3].imshow(child.image) # type: ignore
plt.show()
f.savefig("experiments/reproduction/crossover/"+"crossover_pixel_wise"+".jpg")

#MUTATIONS
    #mutation_polygon
individual = GA_Ini_Pop[-1]
mutation = GA_sol.mutation_Polygon(individual)
f, axarr = plt.subplots(1,3)
f.suptitle("mutation_polygon")
f.supxlabel(str(individual.fitness)+" " +str(mutation.fitness) )
axarr[0].imshow(GA_sol.target)
axarr[1].imshow(individual.image)
axarr[2].imshow(mutation.image)
plt.show()

f.savefig("experiments/reproduction/mutation/"+"mutation_polygon"+".jpg")

    #mutation_pixels
mutation = GA_sol.mutation_Pixels(individual,2000,50)
f, axarr = plt.subplots(1,3)
f.suptitle("mutation_pixels")
f.supxlabel(str(individual.fitness)+" " +str(mutation.fitness) )
axarr[0].imshow(GA_sol.target)
axarr[1].imshow(individual.image)
axarr[2].imshow(mutation.image)
plt.show()
f.savefig("experiments/reproduction/mutation/"+"mutation_pixels"+".jpg")


