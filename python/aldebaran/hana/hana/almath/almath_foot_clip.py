# -*- encoding: UTF-8 -*- 

''' This file shows how to use ALMath library in Python to clip
    NAO footsteps to correct moves (avoiding too big steps and collision).
    Use clipFootStep to clip your desired pose.
'''

import almath

# Various parameters describing step characteristics.
minFootSeparation = 0.088
minStepX = - 0.04
maxStepX = 0.08
maxStepY = 0.16
maxStepTheta = 0.35

def clipFootStepOnGaitConfig(footMove, isLeftSupport):
  ''' Clip the foot move so that it does not exceed the maximum
      size of steps.
      footMove is an almath.Pose2D (x, y, theta position).
      isLeftSupport must be set to True if the move is on the right leg
      (the robot is supporting itself on the left leg).
  '''

  def clipFloat(minValue, maxValue, value):
    ''' Clip value between two extremes. '''
    clipped = value
    if (clipped < minValue):
      clipped = minValue
    if (clipped > maxValue):
      clipped = maxValue
    return clipped

  # Clip X.
  clippedX = clipFloat(minStepX, maxStepX, footMove.x)
  footMove.x = clippedX

  # Clip Y.
  if not isLeftSupport:
    clippedY = clipFloat(minFootSeparation, maxStepY, footMove.y)
  else:
    clippedY = clipFloat(-maxStepY, - minFootSeparation, footMove.y)
  footMove.y = clippedY

  # Clip Theta.
  clippedTheta = clipFloat(-maxStepTheta, maxStepTheta, footMove.theta)
  footMove.theta = clippedTheta


def clipFootStepWithEllipse(footMove):
  ''' Clip the foot move inside an ellipse defined by the foot's dimansions.
      footMove is an almath.Pose2D (x, y, theta position).
  '''

  # Apply an offset to have Y component of foot move centered on 0.
  if (footMove.y < -minFootSeparation):
    footMove.y = footMove.y + minFootSeparation
  elif (footMove.y > minFootSeparation):
    footMove.y = footMove.y - minFootSeparation
  else:
    return

  # Clip the foot move to an ellipse using ALMath method.
  if footMove.x >= 0:
    almath.clipFootWithEllipse(maxStepX, maxStepY - minFootSeparation, footMove)
  else:
    almath.clipFootWithEllipse(minStepX, maxStepY - minFootSeparation, footMove)

  # Correct the previous offset on Y component.
  if footMove.y >=0:
    footMove.y  = footMove.y + minFootSeparation
  else:
    footMove.y = footMove.y - minFootSeparation


def clipFootStepToAvoidCollision(footMove, isLeftSupport):
  ''' Clip the foot move to avoid collision with the other foot.
      footMove is an almath.Pose2D (x, y, theta position).
      isLeftSupport must be set to True if the move is on the right leg
      (the robot is supporting itself on the left leg).
  '''

  # Bounding boxes of NAO's feet.
  rFootBox = [almath.Pose2D(0.11, 0.038, 0.0),    # rFootBoxFL
              almath.Pose2D(0.11, -0.050, 0.0),   # rFootBoxFR
              almath.Pose2D(-0.047, -0.050, 0.0), # rFootBoxRR
              almath.Pose2D(-0.047, 0.038, 0.0)]  # rFootBoxRL
  lFootBox = [almath.Pose2D(0.11, 0.050, 0.0),    # lFootBoxFL
              almath.Pose2D(0.11, -0.038, 0.0),   # lFootBoxFR
              almath.Pose2D(-0.047, -0.038, 0.0), # lFootBoxRR
              almath.Pose2D(-0.047,  0.050, 0.0)] # lFootBoxRL

  # Use ALMath method.
  almath.avoidFootCollision(lFootBox, rFootBox, isLeftSupport, footMove)

def clipFootStep(footMove, isLeftSupport):
  ''' Clipping functions to avoid any warnings and undesired effects
      when sending the footsteps to ALMotion.
      footMove is an almath.Pose2D (x, y, theta position)
      isLeftSupport must be set to True if the move is on the right leg
      (the robot is supporting itself on the left leg).
  '''

  # First clip the foot move with gait config.
  clipFootStepOnGaitConfig(footMove, isLeftSupport)
  # Then clip it on an ellipse.
  clipFootStepWithEllipse(footMove)
  # Then make sure you do not have any collision.
  clipFootStepToAvoidCollision(footMove, isLeftSupport)

