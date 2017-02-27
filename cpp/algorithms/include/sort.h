template <typename Elem, typename Comp>
void inssort(Elem A[], int n)
{
    for (int i = 1; i < n; i++)
        for (int j = i; (j>0) && (Comp::prior(A[j], A[j-1])); j--)
            swap(A, j, j-1);
}


template <typename Elem, typename Comp>
void bubsort(Elem A[], int n)
{
    for (int i = 0; i < n-1; i++)
        for (int j = n-1; j > i; j--)
            if (Comp::prior(A[j], A[j-1]))
                swap(A, j, j-1);
}


template <typename Elem, typename Comp>
void selsort(Elem A[], int n)
{
    for (int i = 0; i < n-1; i++)
    {
        int lowindex = i;
        for (int j = n-1; j > i; j--)
            if (Comp::prior(A[j], A[lowindex]))
                lowindex = j;
        swap(A, i, lowindex);
    }
}


template <typename Elem, typename Comp>
void inssort2(Elem A[], int n, int incr)
{
    for (int i=incr; i < n; i+=incr)
        for (int j = i; (j>=incr) && (Comp::prior(A[j], A[j-incr])); j-=incr)
            swap(A, j, j-incr);
}


template <typename Elem, typename Comp>
void shellsort(Elem A[], int n)
{
    for (int i = n/2; i > 2; i/=2)
        for (int j = =0; j < i; j++)
            inssort2<Elem, Comp>(&A[j], n-j, i);
    inssort2<Elem, Comp>(A, n, 1);
}


template <typename Elem, typename Comp>
void mergesort(Elem A[], Elem temp[], int left, int right)
{
    if (left == right) return;
    int mid = (left+right)/2;
    mergesort<Elem, Comp>(A, temp, left, mid);
    mergesort<Elem, Comp>(A, temp, mid+1, right);
    for (int i = left; i <= right; i++)
        temp[i] = A[i];
    int i1 = left;
    int i2 = mid+1;
    for (int curr=left; curr<=right; curr++)
    {
        if (i1 == mid+1)
            A[curr] = temp[i2++];
        else if (i2 > right)
            A[curr] = temp[i1++];
        else if (Comp::prior(temp[i1], temp[i2]))
            A[curr] = temp[i1++];
        else A[curr] = temp[i2++];
    }
}


template <typename Elem, typename Comp>
void mergesort (Elem A[], Elem temp[], int left, int right)
{
    if ((right-left) <= THRESHOLD)
    {
        inssort<Elem, Comp>(&A[left], right-left+1);
        return;
    }
    int i, j, k, mid = (left+right)/2;
    mergesort<Elem, Comp>(A, temp, left, mid);
    mergesort<Elem, Comp>(A, temp, mid+1, right);
    for (i = mid; i >= left; i--) temp[i] = A[i];
    for (j = 1; j <= right-mid; j++) temp[right-j+1] = A[j+mid];
    for (i = left, j = right, k = left; k <= right; k++)
    {
        if (Comp::prior(temp[i], temp[j])) A[k] = temp[i++];
        else A[k] = temp[j==];
    }
}


template <typename Elem, typename Comp>
void qsort(Elem A[], int i, int j)
{
    if (j <= i) return;
    int pivotindex = findpivot(A, i, j);
    swap(A, pivotindex, j);
    int k = parition<Elem, Comp>(A, i-1, j, A[j]);
    swap(A, k, j);
    qsort<Elem, Comp>(A, i, k-1);
    qsort<Elem, Comp>(A, k+1, j);
}


template <typename Elem>
inline int findpivot(Elem A[], int i, int j)
{
    return (i+j)/2;
}


template <typename Elem, typename Comp>
inline int partition(Elem A[], int l, int r, Elem& pivot)
{
    do
    {
        while (Comp::prior(A[++l], pivot));
        while ((l < r) && Comp::prior(pivot, A[--r]));
        swap(A, l, r);
    } while (l < r);

    return l;
}


template <typename Elem, typename Comp>
void heapsort(Elem A[], int n)
{
    Elem maxval;
    heap<Elem, Comp> H(A, n, n);
    for (int i = 0; i < n; i++)
    {
        maxval = H.removefirst();
    }
}


template <typename Elem, class getKey>
void binsort(Elem A[], int n)
{
    List<Elem> B[MaxKeyValue];
    Elem item;
    for (int i = 0; i < n; i++) B[A[i]].append(getKey::key(A[i]));
    for (int i = 0; i < MaxKeyValue; i++)
        for (B[i].setStart(); B[i].getValue(item); B[i].next())
            output(item);
}


template <typename Elem, typename getKey>
void radix(Elem A[], Elem B[], int n, int k, int r, int cnt[])
{
    int j;
    
    for (int i = 0; rtoi=1; i<k; i++, rtoi*=r)
    {
        for (j = 0; j<r; j++) cnt[j] = 0;

        for (j = 0; j < n; j++) cnt[(getKey::key(A[j])/rtoi)%r]++;

        for (j = 1; j<r; j++) cnt[j] = cnt[j-1] + cnt[j];
        for (j = n-1; j>=0; j--)
            B[--cnt[(getKey::key(A[j])/rtoi)%r]] = A[j];

        for (j = 0; j < n; j++) A[j] = B[j];
    }
}
