#pragma once
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include<fstream>
#include<string.h>
#include <cmath>
#include"haralick_feat.h"
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\ml.hpp>


#define F 12
const double PI = atan(1) * 4;


namespace CppCLRWinformsProjekt {

    using namespace System;
    using namespace System::ComponentModel;
    using namespace System::Collections;
    using namespace System::Windows::Forms;
    using namespace System::Data;
    using namespace System::Drawing;
    using namespace std;
    using namespace cv;
    using namespace cv::ml;
    Ptr<SVM> svm = SVM::create();

    double degrees_to_radians(double y) {
        double radians = (y * PI) / 180; // y not x
        return radians;
    }

    double round(double var)
    {
        // 37.66666 * 100 =3766.66
        // 3766.66 + .5 =3767.16    for rounding off value
        // then type cast to int so value is 3767
        // then divided by 100 so the value converted into 37.67
        double value = (int)(var * 100 + .5);
        return (double)value / 100;
    }

    // GSD is cm/pixel
    double GSD(double z,double sensor_width,double focal_length ,double image_width) {
        
        return ((sensor_width * z * 100) / (focal_length * image_width));

    }

    double xposition(double x, double y, double z, double alpha, double bata)
    {

        double k = z* cos(degrees_to_radians(bata));

        return round(k* sin(degrees_to_radians(bata)));
    }

    double yposition(double x, double y, double z, double alpha, double bata)
    {

        double k = z * cos(degrees_to_radians(alpha));

        return round(k * sin(degrees_to_radians(alpha)));
    }


    // ============================================================================
    //Marginal probabilities as in px = sum on j(p(i, j))
    //                             py = sum on i(p(i, j))
    vector<double> MargProbx(Mat cooc) {
        vector<double> result(cooc.rows, 0.0);
        for (int i = 0; i < cooc.rows; i++)
            for (int j = 0; j < cooc.cols; j++)
                result[i] += cooc.at<double>(i, j);
        return result;
    }

    vector<double> MargProby(Mat cooc) {
        vector<double> result(cooc.cols, 0.0);
        for (int j = 0; j < cooc.cols; j++)
            for (int i = 0; i < cooc.rows; i++)
                result[j] += cooc.at<double>(i, j);
        return result;
    }

    //probsum  := Px+y(k) = sum(p(i,j)) given that i + j = k
    vector<double> ProbSum(Mat cooc) {
        vector<double> result(cooc.rows * 2, 0.0);
        for (int i = 0; i < cooc.rows; i++)
            for (int j = 0; j < cooc.cols; j++)
                result[i + j] += cooc.at<double>(i, j);
        return result;
    }

    //probdiff := Px-y(k) = sum(p(i,j)) given that |i - j| = k
    vector<double> ProbDiff(Mat cooc) {
        vector<double> result(cooc.rows, 0.0);
        for (int i = 0; i < cooc.rows; i++)
            for (int j = 0; j < cooc.cols; j++)
                result[abs(i - j)] += cooc.at<double>(i, j);
        return result;
    }


    /*Features from coocurrence matrix*/
    double HaralickEnergy(Mat cooc) {
        double energy = 0;
        for (int i = 0; i < cooc.rows; i++) {
            for (int j = 0; j < cooc.cols; j++) {
                energy += cooc.at<double>(i, j) * cooc.at<double>(i, j);
            }
        }
        return energy;
    }

    double HaralickEntropy(Mat cooc) {
        double entrop = 0.0;
        for (int i = 0; i < cooc.rows; i++)
            for (int j = 0; j < cooc.cols; j++)
                entrop += cooc.at<double>(i, j) * log(cooc.at<double>(i, j) + EPS);
        return -1 * entrop;
    }

    double HaralickInverseDifference(Mat cooc) {
        double res = 0;
        for (int i = 0; i < cooc.rows; i++)
            for (int j = 0; j < cooc.cols; j++)
                res += cooc.at<double>(i, j) * (1 / (1 + (i - j) * (i - j)));
        return res;
    }

    /*Features from MargProbs */
    double HaralickCorrelation(Mat cooc, vector<double> probx, vector<double> proby) {
        double corr = 0;
        double meanx, meany, stddevx, stddevy;
        meanStd(probx, meanx, stddevx);
        meanStd(proby, meany, stddevy);
        for (int i = 0; i < cooc.rows; i++)
            for (int j = 0; j < cooc.cols; j++)
                corr += (i * j * cooc.at<double>(i, j)) - meanx * meany;
        return corr / (stddevx * stddevy);
    }

    //InfoMeasure1 = HaralickEntropy - HXY1 / max(HX, HY)
    //HXY1 = sum(sum(p(i, j) * log(px(i) * py(j))
    double HaralickInfoMeasure1(Mat cooc, double ent, vector<double> probx, vector<double> proby) {
        double hx = Entropy(probx);
        double hy = Entropy(proby);
        double hxy1 = 0.0;
        for (int i = 0; i < cooc.rows; i++)
            for (int j = 0; j < cooc.cols; j++)
                hxy1 += cooc.at<double>(i, j) * log(probx[i] * proby[j] + EPS);
        hxy1 = -1 * hxy1;

        return (ent - hxy1) / max(hx, hy);

    }

    //InfoMeasure2 = sqrt(1 - exp(-2(HXY2 - HaralickEntropy)))
    //HX2 = sum(sum(px(i) * py(j) * log(px(i) * py(j))
    double HaralickInfoMeasure2(Mat cooc, double ent, vector<double> probx, vector<double> proby) {
        double hxy2 = 0.0;
        for (int i = 0; i < cooc.rows; i++)
            for (int j = 0; j < cooc.cols; j++)
                hxy2 += probx[i] * proby[j] * log(probx[i] * proby[j] + EPS);
        hxy2 = -1 * hxy2;

        return sqrt(1 - exp(-2 * (hxy2 - ent)));
    }

    /*Features from ProbDiff*/
    double HaralickContrast(Mat cooc, vector<double> diff) {
        double contrast = 0.0;
        for (int i = 0; i < diff.size(); i++)
            contrast += i * i * diff[i];
        return contrast;
    }

    double HaralickDiffEntropy(Mat cooc, vector<double> diff) {
        double diffent = 0.0;
        for (int i = 0; i < diff.size(); i++)
            diffent += diff[i] * log(diff[i] + EPS);
        return -1 * diffent;
    }

    double HaralickDiffVariance(Mat cooc, vector<double> diff) {
        double diffvar = 0.0;
        double diffent = HaralickDiffEntropy(cooc, diff);
        for (int i = 0; i < diff.size(); i++)
            diffvar += (i - diffent) * (i - diffent) * diff[i];
        return diffvar;
    }

    /*Features from Probsum*/
    double HaralickSumAverage(Mat cooc, vector<double> sumprob) {
        double sumav = 0.0;
        for (int i = 0; i < sumprob.size(); i++)
            sumav += i * sumprob[i];
        return sumav;
    }

    double HaralickSumEntropy(Mat cooc, vector<double> sumprob) {
        double sument = 0.0;
        for (int i = 0; i < sumprob.size(); i++)
            sument += sumprob[i] * log(sumprob[i] + EPS);
        return -1 * sument;
    }

    double HaralickSumVariance(Mat cooc, vector<double> sumprob) {
        double sumvar = 0.0;
        double sument = HaralickSumEntropy(cooc, sumprob);
        for (int i = 0; i < sumprob.size(); i++)
            sumvar += (i - sument) * (i - sument) * sumprob[i];
        return sumvar;
    }


    Mat MatCooc(Mat img, int N, int deltax, int deltay)
    {
        int atual, vizinho;
        int newi, newj;
        Mat ans = Mat::zeros(N + 1, N + 1, CV_64F);
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                newi = i + deltay;
                newj = j + deltax;
                if (newi < img.rows && newj < img.cols && newj >= 0 && newi >= 0) {
                    atual = (int)img.at<uchar>(i, j);
                    vizinho = (int)img.at<uchar>(newi, newj);
                    ans.at<double>(atual, vizinho) += 1.0;
                }
            }
        }
        return ans / (img.rows * img.cols);
    }

    //Assume tamanho deltax == tamanho deltay 
    Mat MatCoocAdd(Mat img, int N, std::vector<int> deltax, std::vector<int> deltay)
    {
        Mat ans, nextans;
        ans = MatCooc(img, N, deltax[0], deltay[0]);
        for (int i = 1; i < deltax.size(); i++) {
            nextans = MatCooc(img, N, deltax[i], deltay[i]);
            add(ans, nextans, ans);
        }
        return ans;
    }

    void printMat(Mat img) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++)
                printf("%lf ", (double)img.at<double>(i, j));
            printf("\n");
        }
    }


    double* haralik_features(Mat image) {
        int N;
        double min, max;
        static double arr1[F];
        std::vector<int> deltax({ 1 });
        std::vector<int> deltay({ 0 });

        cv::Mat greyMat, colorMat = image;
        cv::cvtColor(colorMat, greyMat, cv::COLOR_BGR2GRAY);

        //Mat img = imread(img, IMREAD_GRAYSCALE);
        Mat ans = MatCoocAdd(greyMat, 255, deltax, deltay);
        std::vector<double> sum = ProbSum(ans);
        std::vector<double> diff = ProbDiff(ans);
        std::vector<double> probx = MargProbx(ans);
        std::vector<double> proby = MargProby(ans);
        double ent = HaralickEntropy(ans);
        double invdiff = HaralickInverseDifference(ans);
        //  cout << "Energy: " << HaralickEnergy(ans) << endl;
        arr1[0] = HaralickEnergy(ans);
        //  cout << arr1[0] << endl;
        //  cout << "Entropy: " << ent << endl;
        arr1[1] = ent;
        //  cout << arr1[1] << endl;
        //  cout << "Inverse Difference: " << invdiff << endl;
        arr1[2] = invdiff;
        //  cout << arr1[2] << endl;
        //  cout << "Correlation: " << HaralickCorrelation(ans, probx, proby) << endl;
        arr1[3] = HaralickCorrelation(ans, probx, proby) / (100000000);
        //  cout << "Info Measure of Correlation 1: " << HaralickInfoMeasure1(ans, ent, probx, proby) << endl;
        arr1[4] = HaralickInfoMeasure1(ans, ent, probx, proby);
        //  cout << "Info Measure of Correlation 2: " << HaralickInfoMeasure2(ans, ent, probx, proby) << endl;
        arr1[5] = HaralickInfoMeasure2(ans, ent, probx, proby);
        //  cout << "Contrast: " << HaralickContrast(ans, diff) << endl;
        arr1[6] = HaralickContrast(ans, diff);
        //  cout << "Difference Entropy: " << HaralickDiffEntropy(ans, diff) << endl;
        arr1[7] = HaralickDiffEntropy(ans, diff);
        //  cout << "Difference Variance: " << HaralickDiffVariance(ans, diff) << endl;
        arr1[8] = HaralickDiffVariance(ans, diff);
        // cout << "Sum Average: " << HaralickSumAverage(ans, sum) << endl;
        arr1[9] = HaralickSumAverage(ans, sum);
        // cout << "Sum Entropy: " << HaralickSumEntropy(ans, sum) << endl;
        arr1[10] = HaralickSumEntropy(ans, sum);
        //cout << "Sum Variance: " << HaralickSumVariance(ans, sum) << endl;
        arr1[11] = HaralickSumVariance(ans, sum);
        return arr1;
    }
    /// <summary>
    /// Zusammenfassung f�r Form1
    /// </summary>
    public ref class Form1 : public System::Windows::Forms::Form
    {
    public:
        Form1(void)
        {
            InitializeComponent();
            //
            //TODO: Konstruktorcode hier hinzuf�gen.
            //
        }

    protected:
        /// <summary>
        /// Verwendete Ressourcen bereinigen.
        /// </summary>
        ~Form1()
        {
            if (components)
            {
                delete components;
            }
        }
    private: System::Windows::Forms::Button^ button1;
    private: System::Windows::Forms::Label^ label1;
    private: System::Windows::Forms::Label^ label2;
    private: System::Windows::Forms::Label^ label3;
    private: System::Windows::Forms::TextBox^ textBox1;
    private: System::Windows::Forms::Label^ label4;
    private: System::Windows::Forms::TextBox^ textBox2;
    private: System::Windows::Forms::TextBox^ textBox3;


    protected:

    private:
        /// <summary>
        /// Erforderliche Designervariable.
        /// </summary>
        System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
        /// <summary>
        /// Erforderliche Methode f�r die Designerunterst�tzung.
        /// Der Inhalt der Methode darf nicht mit dem Code-Editor ge�ndert werden.
        /// </summary>
        void InitializeComponent(void)
        {
            this->button1 = (gcnew System::Windows::Forms::Button());
            this->label1 = (gcnew System::Windows::Forms::Label());
            this->label2 = (gcnew System::Windows::Forms::Label());
            this->label3 = (gcnew System::Windows::Forms::Label());
            this->textBox1 = (gcnew System::Windows::Forms::TextBox());
            this->label4 = (gcnew System::Windows::Forms::Label());
            this->textBox2 = (gcnew System::Windows::Forms::TextBox());
            this->textBox3 = (gcnew System::Windows::Forms::TextBox());
            this->SuspendLayout();
            // 
            // button1
            // 
            this->button1->Location = System::Drawing::Point(96, 189);
            this->button1->Name = L"button1";
            this->button1->Size = System::Drawing::Size(75, 25);
            this->button1->TabIndex = 0;
            this->button1->Text = L"Start Video";
            this->button1->UseVisualStyleBackColor = true;
            this->button1->Click += gcnew System::EventHandler(this, &Form1::button1_Click);
            // 
            // label1
            // 
            this->label1->AutoSize = true;
            this->label1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(0)));
            this->label1->Location = System::Drawing::Point(63, 149);
            this->label1->Name = L"label1";
            this->label1->Size = System::Drawing::Size(144, 15);
            this->label1->TabIndex = 1;
            this->label1->Text = L"Press below to detect";
            this->label1->Click += gcnew System::EventHandler(this, &Form1::label1_Click);
            // 
            // label2
            // 
            this->label2->AutoSize = true;
            this->label2->Location = System::Drawing::Point(-3, 109);
            this->label2->Name = L"label2";
            this->label2->Size = System::Drawing::Size(174, 13);
            this->label2->TabIndex = 3;
            this->label2->Text = L"The coordinates of the object (x,y) :";
            this->label2->Click += gcnew System::EventHandler(this, &Form1::label2_Click);
            // 
            // label3
            // 
            this->label3->AutoSize = true;
            this->label3->Location = System::Drawing::Point(-3, 31);
            this->label3->Name = L"label3";
            this->label3->Size = System::Drawing::Size(159, 13);
            this->label3->TabIndex = 4;
            this->label3->Text = L"Coordinates of the drone (x,y,z) :";
            this->label3->Click += gcnew System::EventHandler(this, &Form1::label3_Click);
            // 
            // textBox1
            // 
            this->textBox1->Location = System::Drawing::Point(159, 28);
            this->textBox1->Name = L"textBox1";
            this->textBox1->Size = System::Drawing::Size(100, 20);
            this->textBox1->TabIndex = 5;
            this->textBox1->TextChanged += gcnew System::EventHandler(this, &Form1::textBox1_TextChanged);
            // 
            // label4
            // 
            this->label4->AutoSize = true;
            this->label4->Location = System::Drawing::Point(2, 72);
            this->label4->Name = L"label4";
            this->label4->Size = System::Drawing::Size(151, 13);
            this->label4->TabIndex = 6;
            this->label4->Text = L"angile of the camera (degree) :";
            this->label4->Click += gcnew System::EventHandler(this, &Form1::label4_Click);
            // 
            // textBox2
            // 
            this->textBox2->Location = System::Drawing::Point(159, 69);
            this->textBox2->Name = L"textBox2";
            this->textBox2->Size = System::Drawing::Size(100, 20);
            this->textBox2->TabIndex = 7;
            this->textBox2->TextChanged += gcnew System::EventHandler(this, &Form1::textBox2_TextChanged);
            // 
            // textBox3
            // 
            this->textBox3->Location = System::Drawing::Point(170, 106);
            this->textBox3->Name = L"textBox3";
            this->textBox3->Size = System::Drawing::Size(89, 20);
            this->textBox3->TabIndex = 8;
            this->textBox3->TextChanged += gcnew System::EventHandler(this, &Form1::textBox3_TextChanged);
            // 
            // Form1
            // 
            this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
            this->ClientSize = System::Drawing::Size(271, 261);
            this->Controls->Add(this->textBox3);
            this->Controls->Add(this->textBox2);
            this->Controls->Add(this->label4);
            this->Controls->Add(this->textBox1);
            this->Controls->Add(this->label3);
            this->Controls->Add(this->label2);
            this->Controls->Add(this->label1);
            this->Controls->Add(this->button1);
            this->Name = L"Form1";
            this->Text = L"Camouflage Detection";
            this->Load += gcnew System::EventHandler(this, &Form1::Form1_Load);
            this->ResumeLayout(false);
            this->PerformLayout();

        }




#pragma endregion
    private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {

        //initial coordinates x,y,z
        double x = 0, y = 0, z = 3, x_ground = 0, y_ground = 0, bata = 0, alpha = 30;
        // drone camera specs
        double sensor_width = 13.2, focal_length = 8.8, image_width = 1280;
        // speed in our video for our simulation speed in m/sec the hight is in meter
        double speed = 0.78, hight = 3;
        int response = 0, flag = 0, temp_i = 0, temp_j = 0, temp, f = 0;
        int labels[3384];
        float trainingData[3384][1];
        float value;
        float sample1[1][1];
        double* p;
        char str[128];
        cv::Mat frame;

        // Set up training data
        //! [setup1]
        //! 

       // cropImage_save();

        std::fstream myfile;
        myfile.open("svm_two_class_train_dense_25.csv");
        int i = 0, ii, j, jj;
        string current_line;

        while (i < 3384) {
            string line;
            getline(myfile, current_line);
            stringstream temp(current_line);
            for (int j = 0; j < 13; j++) {
                getline(temp, line, ',');
                if (!line.empty())
                    value = std::stod(line);
                //cout << x << endl;
                // 

                if (j == 12) {
                    labels[i] = (int)value;
                }
                if (j == 3) {
                    trainingData[i][0] = (float)value;
                }

            }
            i++;
        }


        //! [setup1]
        //! [setup2]
        Mat trainingDataMat(3384, 1, CV_32F, trainingData);
        Mat labelsMat(3384, 1, CV_32SC1, labels);

        //! [setup2]

        // Train the SVM
        //! [init]

        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
        //! [init]
        //! [train]
        svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
        //! [train]




        double footprint = GSD(z, sensor_width, focal_length, image_width) * image_width;

        cv::VideoCapture capture(0);


        if (!capture.isOpened())
        {
            exit(0);
        }

        int frame_count = 0;
        bool should_stop = false;

        while (!should_stop)
        {
            capture >> frame; //get a new frame from the video


            textBox1->Text = "(" + System::Convert::ToString(round(x)) + ", " + System::Convert::ToString(round(y)) + ", " + System::Convert::ToString(round(z)) + ")";


            Mat image = frame.clone();

            if (frame.empty())
            {
                should_stop = true; //we arrived to the end of the video
                continue;
            }

            double sum_row = 0, sum_col = 0, count_detections = 0;
            for (i = 1, ii = 0; i < 10; i++, ii++) {
                
                for (j = 1, jj = 0; j < 10; j++, jj++) {

                    Mat cropped_image = frame(Range(ii * 120, 120 * i), Range(jj * 213, 213 * j));
                    p = haralik_features(cropped_image);
                    sample1[0][0] = (float)p[3];
                    Mat sampleMat(1, 1, CV_32F, sample1);

                    response = svm->predict(sampleMat);

                    if (response == -1) {
                        temp_i = ii * 120;
                        sum_row += ii;
                        sum_col += jj;
                        count_detections++;
                        temp_j = jj * 213;
                        for (int i = temp_i; i < temp_i + 120; i++) {
                            for (int j = temp_j; j < temp_j + 213; j++) {
                                frame.at<Vec3b>(i, j)[2] = 230;

                            }
                        }

                        // for our simulation
                        alpha = 0;
                        bata = 30;

                        //  for our simulation 
                        textBox2->Text = "alpha=" + System::Convert::ToString(round(alpha)) + ", beta=" + System::Convert::ToString(round(bata));

                        // center of mass
                        double center_row = sum_row / count_detections;
                        double center_col = sum_col / count_detections;

                        
                        // generic code
                        x_ground = x+ xposition(x, y, z, alpha, bata)+((9-center_row)*213*0.234/100);
                        y_ground = y + yposition(x, y, z, alpha, bata) + ((9 - center_col) * 120 * 0.234 / 100);


                        textBox3->Text = "(" + System::Convert::ToString(round(x_ground)) + ", " + System::Convert::ToString(y_ground) + ")";
                    }
                }
            }

            //display frames
            namedWindow("original", 0);

            imshow("original", image);
            resizeWindow("original", 480, 300);


            namedWindow("Detection", 0);

            imshow("Detection", frame);
            resizeWindow("Detection", 480, 300);

            //wait 20 ms between successive frames and break the loop if key q is pressed
            int key = waitKey(20);
            if (key == 'q')
            {
                cout << "q key is pressed by the user. Stopping the video" << endl;
                break;
            }

            x += speed/24;// speed is m/sec divide by 24 frame per seconde

            ++frame_count;

        }


        capture.release();
        destroyAllWindows();
    }
    private: System::Void label1_Click(System::Object^ sender, System::EventArgs^ e) {
    }
    private: System::Void Form1_Load(System::Object^ sender, System::EventArgs^ e) {
    }
    private: System::Void label2_Click(System::Object^ sender, System::EventArgs^ e) {
    }

    private: System::Void label3_Click(System::Object^ sender, System::EventArgs^ e) {


    }
    private: System::Void label4_Click(System::Object^ sender, System::EventArgs^ e) {

    }
    private: System::Void textBox1_TextChanged(System::Object^ sender, System::EventArgs^ e) {

    }
    private: System::Void textBox2_TextChanged(System::Object^ sender, System::EventArgs^ e) {
    }
    private: System::Void textBox3_TextChanged(System::Object^ sender, System::EventArgs^ e) {
    }
    private: System::Void label5_Click(System::Object^ sender, System::EventArgs^ e) {
        //string ^ in = textBox1->Text;

    }

    };

}


