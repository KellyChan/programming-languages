/*

    Shape 
     +
     +----> Circle  -----> Simley
     +----> Triangle

*/


class Shape {

    public:
        virtual Point center() const = 0;  // pure virtual
        virtual void move(Point to) = 0;

        virtual void draw() const = 0;  // draw on current Canvas
        virtual void rotate(int angle) = 0;

        virtual ~Shape(){}    // destructor
}


void rotate_all(vector<Shape*>& v, int angle)    // rotate v's elementss by angle degrees
{
    for (auto p: v)
    {
        p->rotate(angle);
    }
}



class Circle : public Shape
{

    public:
        Circle(Point p, int rr);  // constructor

        Point center() const {return x;}
        void move(Point to) {x = to;}
        void draw() const;
        void rotate(int) {}    // nice simple algorithm

    private:
        Point x:  // center
        int r;    // radius
}


class Smiley : public Circle 
{
    public:
        Smiley(Point p, int r): Circle(p, r), mouth {nullptr} {}

        ~Smiley()
        {
            delete mouth;
            for (auto p: eyes)
                delete p;
        }

        void move(Point to) override;
        void draw() const override;
        void rotate(int) override;

        void add_eye(Shape* s) {eyes.push_back(s);}
        void set_mouth(Shape* s);
        virtual void wink(int i);  // wink eye number

    private:
        vector<Shape*> eyes;  // usually two eyes
        Shape* mouth;
}


void Smiley::draw()
{
    Circle::draw();
    for (auto p: eyes)
        p->draw();
        mouth->draw();
}


enum class Kind { circle, triangle, smiley };

Shape* read_shape(stream& is)  // read shape descriptions from input stream is
{
    // read shape header from is and find its Kind k ...
    switch(k)
    {
        case Kind::circle:
            // read circle data {Point, int} into p and r
            return new Circle(p, r);
        case Kind::triangle:
            // read triangle data {Point, Point, Point} into p1, p2, and p3
            return new Triangle(p1, p2, p3);
        case Kind::smiley:
            // read smiley data {Point, int, Shape, Shape, Shape} into p, r, e1, e2, and m
            Smiley* ps = new Smiley{p, r};
            ps->add_eye(e1);
            ps->add_eye(e2);
            ps->set_mouth(m);
            return ps;     
    }
}


void user()
{
    std::vector<Shape*> v;
    while(cin)
    {
        v.push_back(read_shape(cin));
    }
    draw_all(v);  // call draw() for each element
    rotate_all(v, 45);  // call rotate(45) for each element
    for (auto p: v)
        delete p;
}
