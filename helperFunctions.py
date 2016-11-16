import math, random


def mergeSort(solutions):
    if len(solutions) > 1 :
        mid         = len(solutions)//2
        lefthalf    = solutions[:mid]
        righthalf   = solutions[mid:]
        mergeSort(lefthalf)
        mergeSort(righthalf)

        i = 0
        j = 0
        k = 0
        while i<len(lefthalf) and j < len(righthalf):
            if(lefthalf[i].fitness < righthalf[j].fitness ):
                solutions[k] = lefthalf[i]
                i+=1
            else :
                solutions[k] = righthalf[j]
                j+=1
            k+=1

        while i < len(lefthalf):
            solutions[k] = lefthalf[i]
            i+=1
            k+=1

        while j < len(righthalf):
            solutions[k] = righthalf[j]
            j+=1
            k+=1

def dist(location1, location2):
    return math.sqrt( pow(location1.position.x - location2.position.x, 2) + pow(location1.position.y - location2.position.y, 2) )
