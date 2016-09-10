package bobsrockets {
  package navigation {
    // In package bobsrockets.navigation
    class Navigation

    package tests {
      // In package bobsrockets.navigation.tests
      class NavigatorSuite
    }
  }
}

//------------------------------------------------------------

package bobsrockets {
  
  package navigation {
    class Navigator {
      // No need to say bobsrockets.navigation.StartMap
      val map = new StarMap
    }
    
    class StarMap
  }

  class Ship {
    // No need to say bobsrockets.navigation.Navigator
    val nav = new Navigation.Navigator
  }

  package fleets {
    class Fleet {
      // No need to say bobsrockets.Ship
      def addShip() {new Ship}
    }
  }
}
