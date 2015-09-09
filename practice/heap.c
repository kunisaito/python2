#include <stdio.h>
#define MAX_N 100000
int heap[MAX_N],sz = 0;

void push(int x){
  int i = sz ++;
  while(i > 0){
    int p = (i-1)/2;
    if (heap[p]<=x)break;
    heap[i] = heap[p];
    i = p;
  }
  heap[i] = x;
}

int pop(){
  int pop_value = heap[0];
  int top_value = heap[--sz];
  int i=0;
  while(i*2+1<sz){
    int a = i*2 +1;
    int b = i*2 +2;

    if(b<sz && heap[a] > heap[b]){
      a = b;
	}
    if(heap[a] >= top_value){
      break;
    }

    heap[i] = heap[a];
    i = a;
  }
  heap[i] = top_value;
  return pop_value;
}
int main(void){
  push(3);
  push(2);
  push(6);
  push(7);
  push(0);
  push(4);
  push(5);
  printf("%d\n",pop());
  printf("%d\n",pop());

}
    
      
  
