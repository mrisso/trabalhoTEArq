int ompMultMatrix(double** a, double** b, double** c, int width)
{
   int i,j,k;


   for (i=0; i<width; i=i+1){
      for (j=0; j<width; j=j+1){
         c[i][j]=0;
         for (k=0; k<width; k=k+1){
            c[i][j]=(c[i][j])+((a[i][k])*(b[k][j]));
         }
      }
   }
   return 0;
}
