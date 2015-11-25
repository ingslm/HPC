#include <iostream>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iomanip>
using namespace std;
void merge(int, int, int, int*);

void merge_sort (int array[], int size)
{
    int temp[size];
    int mid, i;
    if (size < 2) {
        return;
    } 
    else {
        mid = size / 2;
        merge_sort(array, mid);
        merge_sort(array + mid, size - mid);
        merge (array, mid, array + mid, size - mid, temp);
        for (i = 0; i < size; i++) {
            array[i] = temp[i];
        }
    }
}

int  merge  (int list1[ ] , int size1 , int list2[ ] , int size2 , int list3[ ])
{
    int i1, i2, i3;
    if (size1+size2 > size) {
        return false;
    }
    i1 = 0; i2 = 0; i3 = 0;
    /* while both lists are non-empty */
    while (i1 < size1 && i2 < size2) {
        if (list1[i1] < list2[i2]) {
            list3[i3++] = list1[i1++];
        } 
        else {
            list3[i3++] = list2[i2++];
        }
    }
    while (i1 < size1) {   
        /* copy remainder of list1 */
        list3[i3++] = list1[i1++];
    }
    while (i2 < size2) { 
        /* copy remainder of list2 */
        list3[i3++] = list2[i2++];
    }
    return true;
}

int main() 
{
    int a[] = {2, 42, 3, 7, 1};
    merge_sort(0, 4, a);
    for (int i = 0; i<=4 ; i++)
        cout << a[i] << endl;
    return (0);

}
