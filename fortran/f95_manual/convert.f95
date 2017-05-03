module convertT
  implicit none

  real, parameter, private :: factor = 0.55555556
  integer, parameter, private :: offset = 32

  contains

    function CtoF (TinC) result (TinF)
      ! function result
      real :: TinF

      ! dummy argument
      real, intent (in) :: TinC

    end function CtoF


    function FtoC (TinF) result (TinC)
      ! function result
      real :: TinC

      ! dummy arugment
      real :: TinF

      TinC = (TinF - offset) * factor

    end function FtoC

end module convertT


program convert_temperature

  use convertT
  implicit none

  print*, "20 Celcius = ", CtoF (20.0), " Fahrenheit"
  print*, "100 Fahrenheit = ", FtoC (100.0), " Celcius"
end program convert_temperature
