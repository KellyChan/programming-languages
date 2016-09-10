import org.scalatest.Suite
import Element.elem

class ElementSuite extends Suite {
  
  def testUniformElement() {
    val ele = elem('x', 2, 3)
    assert(ele.width == 2)
  }
}

//--------------------------------------------------------

import org.scalatest.FunSuite
import Element.elem

class ElementSuite extends FunSuite {
  test("elem result should have passed width") {
    val ele = elem('x', 2, 3)
    assert(ele.width == 2)
  }
}

//--------------------------------------------------------

import junit.frameework.TestCase
import junit.framework.Assert.assertEquals
import junit.framework.Assert.fail
import Element.elem

class ElementTestCase extends TestCase {

  def testUniformElement() {
    val ele = elem('x', 2, 3)
    assertEquals(2, ele.width)
    assertEquals(3, ele.height)
    try {
      elem('x', -2, 3)
      fail()
    }
    catch {
      case e: IllegalArgumentException => // expected
    }
  }
}

//--------------------------------------------------------

import org.scalatest.junit.JUnit3Suite
import Element.elem

class ElementSuite extends Junit3Suite {
  def testUniformElement() {
    val ele = elem('x', 2, 3)
    assert(ele.width == 2)
    expect(3) { ele.height }
    intercept[IllegalArgumentException] {
      elem('x', -2, 3) 
    }
  }
}
