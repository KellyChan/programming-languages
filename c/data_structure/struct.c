typedef struct address_s
{
  char * street;
  char * city;
  char * region;
  char * country;
  char * postal_code;
} ADDRESS_t, *ADDRESS_p_t;


struct address_s
{
  char * street;
  char * city;
  char * region;
  char * country;
  char * postal_code;
};



// static void print_address (ADDRESS_p_t address_info);
static void print_address (struct address_s * address_info);
static void print_an_address (void)
{
  struct address_s address;

  address.street = "1823 23rd Ave NE";
  address.city = "Seattle";
  address.region = "WA";
  address.postal_code = "98023";
  print_address (&address);
}


int main ()
{
  print_an_address();

  return 0;
}
