ALL: a2

a2: a2.c
	mpicc -Wall a2.c -lm -lpthread -o mpiOut

run:
	mpirun -oversubscribe -np 17 mpiOut 4 4

clean:
	/bin/rm -f mpiOut *.o
	/bin/rm -f Log_File.txt *.o