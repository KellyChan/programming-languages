recursive function nfactorial (n) result (fac)
! computes the factorial of n (n!)

  implicit none

  ! function result
  integer :: fac

  ! dummy arguments
  integer, intent (in) :: n

  select case (n)
    case (0:1)
      fac = 1
    case default
      fac = n * nfactorial (n-1)
  end select

end function nfactorial
