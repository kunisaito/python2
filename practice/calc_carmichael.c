#include <stdio.h>

typedef long long ll;

ll mod_pow(ll x, ll n, ll mod){
  ll res = 1;
  while(n>0){
    if(n & 1) res = res * x % mod;
    x = x * x % mod;
    n >>= 1;
    printf("%d\n",n);
    printf("x %d\n",x);
    printf("res %d\n",res);
  }
  return res;
}
int main(void){
  ll x = 3;
  ll n = 6;
  ll mod = 5;
  int res;
  res = mod_pow(x,n,mod);
  printf("%d\n",res);

  return 0;
}
