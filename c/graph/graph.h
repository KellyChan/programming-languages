typedef struct edgeTag
{
  struct vertexTag *connectsTo;
  struct edgeTag *next;
} edgeT;


typedef struct vectexTag
{
  graphElementT element;
  int visitied;
  struct edgeTag *edges;
  struct vertexTag *next;
} vertexT;

typedef struct graphCDT
{
  vertexT *vertices;
} graphCDT;
