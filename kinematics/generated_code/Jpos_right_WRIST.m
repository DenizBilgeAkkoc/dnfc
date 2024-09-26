% WRIST POSITION JACOBIAN
function J = Jpos_right_WRIST(t1, t2, a1, a2, a3, a4, a5, a6, a7, g)
    J(1,1) = 0.195*cos(t1) - 0.285*sin(t1)*sin(t2) + 0.25*cos(a4)*(sin(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1)) + cos(a2)*cos(t1)) - 0.25*sin(a4)*(cos(a3)*(sin(a2)*cos(t1) - 1.0*cos(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1))) + sin(a3)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1))) + 0.25*sin(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1)) + 0.25*cos(a2)*cos(t1) + 0.02*cos(a1)*sin(t1)*sin(t2) - 0.02*sin(a1)*cos(t2)*sin(t1);
    J(1,2) = 0.285*cos(t1)*cos(t2) - 0.25*sin(a4)*(sin(a3)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2)) + cos(a2)*cos(a3)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2))) - 0.25*sin(a2)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2)) - 0.25*cos(a4)*sin(a2)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2)) - 0.02*cos(a1)*cos(t1)*cos(t2) - 0.02*sin(a1)*cos(t1)*sin(t2);
    J(1,3) = 0.25*sin(a4)*(sin(a3)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2)) + cos(a2)*cos(a3)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2))) + 0.25*sin(a2)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2)) + 0.25*cos(a4)*sin(a2)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2)) + 0.02*cos(a1)*cos(t1)*cos(t2) + 0.02*sin(a1)*cos(t1)*sin(t2);
    J(1,4) = - 0.25*sin(a2)*sin(t1) - 0.25*cos(a4)*(sin(a2)*sin(t1) + cos(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2))) - 0.25*cos(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2)) - 0.25*cos(a3)*sin(a4)*(cos(a2)*sin(t1) - 1.0*sin(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2)));
    J(1,5) = 0.25*sin(a4)*(sin(a3)*(sin(a2)*sin(t1) + cos(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2))) + cos(a3)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2)));
    J(1,6) = - 0.25*cos(a4)*(1.0*cos(a3)*(sin(a2)*sin(t1) + cos(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2))) - sin(a3)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2))) - 0.25*sin(a4)*(cos(a2)*sin(t1) - 1.0*sin(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2)));
    J(1,7) = 0.0;
    J(1,8) = 0.0;
    J(1,9) = 0.0;
    J(1,10) = 0.0;
    J(2,1) = 0.195*sin(t1) + 0.25*cos(a2)*sin(t1) + 0.285*cos(t1)*sin(t2) - 0.25*sin(a4)*(1.0*cos(a3)*(sin(a2)*sin(t1) + cos(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2))) - sin(a3)*(cos(a1)*cos(t1)*cos(t2) + sin(a1)*cos(t1)*sin(t2))) + 0.25*cos(a4)*(cos(a2)*sin(t1) - 1.0*sin(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2))) - 0.25*sin(a2)*(cos(a1)*cos(t1)*sin(t2) - 1.0*sin(a1)*cos(t1)*cos(t2)) - 0.02*cos(a1)*cos(t1)*sin(t2) + 0.02*sin(a1)*cos(t1)*cos(t2);
    J(2,2) = 0.285*cos(t2)*sin(t1) - 0.25*sin(a2)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1)) - 0.25*sin(a4)*(sin(a3)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1)) + cos(a2)*cos(a3)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1))) - 0.02*sin(a1)*sin(t1)*sin(t2) - 0.25*cos(a4)*sin(a2)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1)) - 0.02*cos(a1)*cos(t2)*sin(t1);
    J(2,3) = 0.25*sin(a2)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1)) + 0.25*sin(a4)*(sin(a3)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1)) + cos(a2)*cos(a3)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1))) + 0.02*sin(a1)*sin(t1)*sin(t2) + 0.25*cos(a4)*sin(a2)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1)) + 0.02*cos(a1)*cos(t2)*sin(t1);
    J(2,4) = 0.25*sin(a2)*cos(t1) + 0.25*cos(a4)*(sin(a2)*cos(t1) - 1.0*cos(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1))) - 0.25*cos(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1)) + 0.25*cos(a3)*sin(a4)*(sin(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1)) + cos(a2)*cos(t1));
    J(2,5) = 0.25*sin(a4)*(1.0*cos(a3)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1)) - sin(a3)*(sin(a2)*cos(t1) - 1.0*cos(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1))));
    J(2,6) = 0.25*sin(a4)*(sin(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1)) + cos(a2)*cos(t1)) + 0.25*cos(a4)*(cos(a3)*(sin(a2)*cos(t1) - 1.0*cos(a2)*(cos(a1)*sin(t1)*sin(t2) - 1.0*sin(a1)*cos(t2)*sin(t1))) + sin(a3)*(sin(a1)*sin(t1)*sin(t2) + cos(a1)*cos(t2)*sin(t1)));
    J(2,7) = 0.0;
    J(2,8) = 0.0;
    J(2,9) = 0.0;
    J(2,10) = 0.0;
    J(3,1) = 0.0;
    J(3,2) = 0.02*cos(a1)*sin(t2) - 0.285*sin(t2) - 0.02*sin(a1)*cos(t2) - 0.25*sin(a4)*(sin(a3)*(sin(a1)*sin(t2) + cos(a1)*cos(t2)) - 1.0*cos(a2)*cos(a3)*(cos(a1)*sin(t2) - 1.0*sin(a1)*cos(t2))) + 0.25*sin(a2)*(cos(a1)*sin(t2) - 1.0*sin(a1)*cos(t2)) + 0.25*cos(a4)*sin(a2)*(cos(a1)*sin(t2) - 1.0*sin(a1)*cos(t2));
    J(3,3) = 0.02*sin(a1)*cos(t2) - 0.02*cos(a1)*sin(t2) + 0.25*sin(a4)*(sin(a3)*(sin(a1)*sin(t2) + cos(a1)*cos(t2)) - 1.0*cos(a2)*cos(a3)*(cos(a1)*sin(t2) - 1.0*sin(a1)*cos(t2))) - 0.25*sin(a2)*(cos(a1)*sin(t2) - 1.0*sin(a1)*cos(t2)) - 0.25*cos(a4)*sin(a2)*(cos(a1)*sin(t2) - 1.0*sin(a1)*cos(t2));
    J(3,4) = 0.25*cos(a3)*sin(a2)*sin(a4)*(sin(a1)*sin(t2) + cos(a1)*cos(t2)) - 0.25*cos(a2)*cos(a4)*(sin(a1)*sin(t2) + cos(a1)*cos(t2)) - 0.25*cos(a2)*(sin(a1)*sin(t2) + cos(a1)*cos(t2));
    J(3,5) = -0.25*sin(a4)*(cos(a3)*(cos(a1)*sin(t2) - 1.0*sin(a1)*cos(t2)) - 1.0*cos(a2)*sin(a3)*(sin(a1)*sin(t2) + cos(a1)*cos(t2)));
    J(3,6) = 0.25*sin(a2)*sin(a4)*(sin(a1)*sin(t2) + cos(a1)*cos(t2)) - 0.25*cos(a4)*(sin(a3)*(cos(a1)*sin(t2) - 1.0*sin(a1)*cos(t2)) + cos(a2)*cos(a3)*(sin(a1)*sin(t2) + cos(a1)*cos(t2)));
    J(3,7) = 0.0;
    J(3,8) = 0.0;
    J(3,9) = 0.0;
    J(3,10) = 0.0;
%end of Jacobian computation

