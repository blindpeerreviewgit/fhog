#define DLIB_PNG_SUPPORT
#define DLIB_JPEG_SUPPORT

#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

namespace dlib {
template <
    typename pyramid_type,
    typename image_array_type
    >
void upsample_image_dataset_if_small (
    image_array_type& images,
    std::vector<std::vector<rectangle> >& objects
)
{
    // make sure requires clause is not broken
    DLIB_ASSERT( images.size() == objects.size(),
        "\t void upsample_image_dataset()"
        << "\n\t Invalid inputs were given to this function."
        << "\n\t images.size():   " << images.size() 
        << "\n\t objects.size():  " << objects.size() 
        );

    typename image_array_type::value_type temp;
    pyramid_type pyr;
    for (unsigned long i = 0; i < images.size(); ++i)
    {
        if (images[i].nc() == 320 && images[i].nr() == 240)
        {
            pyramid_up(images[i], temp, pyr);
            swap(temp, images[i]);
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                objects[i][j] = pyr.rect_up(objects[i][j]);
            }
        }
    }
}
}

int main(int argc, char** argv)
{  

    try
    {
        if (argc != 3)
        {
            cout << "Please use :   ./prog dir_training dir_testing" << endl;
            cout << endl;
            return 0;
        }
        const std::string robots_directory_training = argv[1];
        const std::string robots_directory_testing = argv[2];

        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<rectangle> > robot_boxes_train, robot_boxes_test;

        // **************** Load the data. ********************
        load_image_dataset(images_train, robot_boxes_train, robots_directory_training+"/training.xml");
        load_image_dataset(images_test, robot_boxes_test, robots_directory_testing+"/testing.xml");


	// **************** Pre-processing. ********************
        // This is optional but for
        // this training data it improves the results.  The first things we do is
        // increase the size of the images by a factor of two.  We do this
        // because it will allow us to detect smaller robots than otherwise would
        // be practical (since the robots are all now twice as big).  Note that,
        // in addition to resizing the images, these functions also make the
        // appropriate adjustments to the robot boxes so that they still fall on
        // top of the robots after the images are resized.
        // upsample_image_dataset<pyramid_down<2> >(images_train, robot_boxes_train);
        // upsample_image_dataset<pyramid_down<2> >(images_test,  robot_boxes_test);
        // upsample_image_dataset<pyramid_down<3> >(images_train, robot_boxes_train);
        // upsample_image_dataset<pyramid_down<3> >(images_test,  robot_boxes_test);

        upsample_image_dataset_if_small<pyramid_down<2> >(images_train, robot_boxes_train);
        upsample_image_dataset_if_small<pyramid_down<2> >(images_test,  robot_boxes_test);

        //upsample_image_dataset<pyramid_down<2> >(images_train, robot_boxes_train);

        // Since robots are generally left-right symmetric we can increase
        // our training dataset by adding mirrored versions of each image back
        // into images_train.  So this next step doubles the size of our
        // training dataset.  Again, this is obviously optional but is useful in
        // many object detection tasks.
        add_image_left_right_flips(images_train, robot_boxes_train);
        cout << "num training images: " << images_train.size() << endl;
        cout << "num testing images:  " << images_test.size() << endl;


	// **************** Training. ********************
        // Finally we get to the training code.  dlib contains a number of
        // object detectors.  This typedef tells it that you want to use the one
        // based on Felzenszwalb's version of the Histogram of Oriented
        // Gradients (commonly called HOG) detector.  The 6 means that you want
        // it to use an image pyramid that downsamples the image at a ratio of
        // 5/6.  Recall that HOG detectors work by creating an image pyramid and
        // then running the detector over each pyramid level in a sliding window
        // fashion.   
        typedef scan_fhog_pyramid<pyramid_down<5> > image_scanner_type; 
        image_scanner_type scanner;

        // >>>>>>>>>>>> Step 2 : Grid search :: Best config tested (accuracy = 0.924444): hog_window_size =(40,60); Cost=16; epsilon=0.0078125; eps=0.00390625
        // >>>>>>>>>>>> Step 2 : Grid search :: Best config tested (accuracy = 0.93184): hog_window_size =(40,60); Cost=16; epsilon=0.015625; eps=0.015625


        // The sliding window detector will be 80 pixels wide and 80 pixels tall.
        scanner.set_detection_window_size(40, 60); 

        structural_object_detection_trainer<image_scanner_type> trainer(scanner);

        // training results: 0.973684 0.973684 0.969373 
        // testing results:  0.954023 0.897297 0.892555 
        //Step 3 : Random search :: Best config tested (accuracy = 0.892555): hog_window_size =(40,60); Cost=267.438; epsilon=0.0367132; eps=0.118778

        // training results: 0.994681 0.984211 0.984071 
        // testing results:   0.97006 0.875676 0.871019 

        // Step 3 : Random search :: Best config tested (accuracy = 0.871019): hog_window_size =(40,60); Cost=30.1189; epsilon=0.0970016; eps=0.0748489


        // 4/5
        // training results: 0.978723 0.968421 0.962119 
        // testing results:  0.948276 0.891892 0.888203 

        // Step 3 : Random search :: Best config tested (accuracy = 0.888203): hog_window_size =(40,60); Cost=28.3241; epsilon=0.0301585; eps=0.00570746

        // training results: 0.994681 0.984211 0.981999 
        // testing results:  0.948571 0.897297 0.893647 

        // Step 3 : Random search :: Best config tested (accuracy = 0.893647): hog_window_size =(40,60); Cost=29.8787; epsilon=0.0580168; eps=0.00738214




        // Set this to the number of processing cores on your machine.
        trainer.set_num_threads(1); //8);  

        // The trainer is a kind of support vector machine and therefore has the usual SVM
        // C parameter.  In general, a bigger C encourages it to fit the training data
        // better but might lead to overfitting.  You must find the best C value
        // empirically by checking how well the trained detector works on a test set of
        // images you haven't trained on.  Don't just leave the value set at 1.  Try a few
        // different C values and see what works best for your data.
        //trainer.set_c(32);
        //trainer.set_c(267.438);
        //trainer.set_c(30.1189);
        //trainer.set_c(30);
        trainer.set_c(29.8787);
        //trainer.set_c(64);
        // We can tell the trainer to print it's progress to the console if we want.  
        trainer.be_verbose();
        // The trainer will run until the "risk gap" is less than 0.01.  Smaller values
        // make the trainer solve the SVM optimization problem more accurately but will
        // take longer to train.  For most problems a value in the range of 0.1 to 0.01 is
        // plenty accurate.  Also, when in verbose mode the risk gap is printed on each
        // iteration so you can see how close it is to finishing the training.  
        //trainer.set_epsilon(0.00390625);
        //trainer.set_epsilon(0.0367132);
        //trainer.set_epsilon(0.0970016);
        //trainer.set_epsilon(0.1);
        trainer.set_epsilon(0.058);
        //trainer.set_epsilon(0.1);
        
        

	//trainer.set_match_eps(0.00390625);
        //trainer.set_match_eps(0.118778);
        //trainer.set_match_eps(0.0748489);
        trainer.set_match_eps(0.00738);

		//remove_unobtainable_rectangles(trainer, images_train, robot_boxes_train);
        
        


        // Now we run the trainer.
        object_detector<image_scanner_type> detector = trainer.train(images_train, robot_boxes_train);



	// **************** Testing. ********************

        std::cerr << "detector.num_detectors() = " << detector.num_detectors() << "\n";

        // on the training data.
        // auto detect_t0 = std::chrono::high_resolution_clock::now();

        cout << "training results (precision, recall, average precision): " << test_object_detection_function(detector, images_train, robot_boxes_train) << endl;

        // auto detect_t1 = std::chrono::high_resolution_clock::now();

        // on the testing images (to avoid overfitting).
        cout << "testing results (precision, recall, average precision):  " << test_object_detection_function(detector, images_test, robot_boxes_test) << endl;

        // auto detect_t2 = std::chrono::high_resolution_clock::now();

        cout << "detector.num_detectors() = " << detector.num_detectors() << "\n";

        cout << "num filters: "<< num_separable_filters(detector) << endl;
        // You can also control how many filters there are by explicitly thresholding the
        // singular values of the filters like this:
        // detector = threshold_filter_singular_values(detector,0.6);
        // cout << "remaining filters: "<< num_separable_filters(detector) << endl;

        // auto detect_t3 = std::chrono::high_resolution_clock::now();

        // cout << "training results (precision, recall, average precision): " << test_object_detection_function(detector, images_train, robot_boxes_train) << endl;

        // auto detect_t4 = std::chrono::high_resolution_clock::now();

        // cout << "testing results (precision, recall, average precision):  " << test_object_detection_function(detector, images_test, robot_boxes_test) << endl;

        // auto detect_t5 = std::chrono::high_resolution_clock::now();

        // auto detectTime_train1 = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(detect_t1-detect_t0).count();
        // auto detectTime_test1 = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(detect_t2-detect_t1).count();
        // auto detectTime_train2 = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(detect_t4-detect_t3).count();
        // auto detectTime_test2 = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(detect_t5-detect_t4).count();

        // std::cerr << "initial detection time -- train: " << detectTime_train1 << " test: " << detectTime_test1 << "\n";
        // std::cerr << "optimized detection time -- train: " << detectTime_train2 << " test: " << detectTime_test2 << "\n";

	// **************** For fun. ********************

        // "sticks" visualization of a learned HOG detector.
        image_window hogwin(draw_fhog(detector), "Learned fHOG detector");

        // Display the testing images on the screen and
        // show the output of the robot detector overlaid on each image.
        image_window win; 
        for (unsigned long i = 0; i < images_test.size(); ++i)
        {
            // Run the detector and get the robot detections.
            auto detect_t0 = std::chrono::high_resolution_clock::now();
            std::vector<rectangle> dets = detector(images_test[i]);
            auto detect_t1 = std::chrono::high_resolution_clock::now();
            
            win.clear_overlay();
            win.set_image(images_test[i]);
            win.add_overlay(dets, rgb_pixel(255,0,0));
            
            auto detectTime = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(detect_t1-detect_t0).count();
            cout << "full detection time:  " << detectTime << endl;
            
            cout << "Hit enter to process the next image..." << endl;
            cin.get();
        }

        // image_window win; 
        // for (unsigned long i = 0; i < images_train.size(); ++i)
        // {
        //     // Run the detector and get the robot detections.
        //     std::vector<rectangle> dets = detector(images_train[i]);
        //     win.clear_overlay();
        //     win.set_image(images_train[i]);
        //     win.add_overlay(dets, rgb_pixel(255,0,0));
        //     cout << "Hit enter to process the next image..." << endl;
        //     cin.get();
        // }


        // Save your detector to disk
        serialize("robot_detector.svm") << detector;

        // Recall it
        object_detector<image_scanner_type> detector2;
        deserialize("robot_detector.svm") >> detector2;




        // Now let's talk about some optional features of this training tool as well as some
        // important points you should understand.
        //
        // The first thing that should be pointed out is that, since this is a sliding
        // window classifier, it can't output an arbitrary rectangle as a detection.  In
        // this example our sliding window is 80 by 80 pixels and is run over an image
        // pyramid.  This means that it can only output detections that are at least 80 by
        // 80 pixels in size (recall that this is why we upsampled the images after loading
        // them).  It also means that the aspect ratio of the outputs is 1.  So if,
        // for example, you had a box in your training data that was 200 pixels by 10
        // pixels then it would simply be impossible for the detector to learn to detect
        // it.  Similarly, if you had a really small box it would be unable to learn to
        // detect it.  
        //
        // So the training code performs an input validation check on the training data and
        // will throw an exception if it detects any boxes that are impossible to detect
        // given your setting of scanning window size and image pyramid resolution.  You
        // can use a statement like:
        //   remove_unobtainable_rectangles(trainer, images_train, robot_boxes_train)
        // to automatically discard these impossible boxes from your training dataset
        // before running the trainer.  This will avoid getting the "impossible box"
        // exception.  However, I would recommend you be careful that you are not throwing
        // away truth boxes you really care about.  The remove_unobtainable_rectangles()
        // will return the set of removed rectangles so you can visually inspect them and
        // make sure you are OK that they are being removed. 
        // 
        // Next, note that any location in the images not marked with a truth box is
        // implicitly treated as a negative example.  This means that when creating
        // training data it is critical that you label all the objects you want to detect.
        // So for example, if you are making a robot detector then you must mark all the
        // robots in each image.  However, sometimes there are objects in images you are
        // unsure about or simply don't care if the detector identifies or not.  For these
        // objects you can pass in a set of "ignore boxes" as a third argument to the
        // trainer.train() function.  The trainer will simply disregard any detections that
        // happen to hit these boxes.  
        //
        // Another useful thing you can do is evaluate multiple HOG detectors together. The
        // benefit of this is increased testing speed since it avoids recomputing the HOG
        // features for each run of the detector.  You do this by storing your detectors
        // into a std::vector and then invoking evaluate_detectors() like so:
        std::vector<object_detector<image_scanner_type> > my_detectors;
        my_detectors.push_back(detector);
        std::vector<rectangle> dets = evaluate_detectors(my_detectors, images_train[0]); 
        //
        //
        // Finally, you can add a nuclear norm regularizer to the SVM trainer.  Doing has
        // two benefits.  First, it can cause the learned HOG detector to be composed of
        // separable filters and therefore makes it execute faster when detecting objects.
        // It can also help with generalization since it tends to make the learned HOG
        // filters smoother.  To enable this option you call the following function before
        // you create the trainer object:
        //    scanner.set_nuclear_norm_regularization_strength(1.0);
        // The argument determines how important it is to have a small nuclear norm.  A
        // bigger regularization strength means it is more important.  The smaller the
        // nuclear norm the smoother and faster the learned HOG filters will be, but if the
        // regularization strength value is too large then the SVM will not fit the data
        // well.  This is analogous to giving a C value that is too small.
        //
        // You can see how many separable filters are inside your detector like so:
        // cout << "num filters: "<< num_separable_filters(detector) << endl;
        // You can also control how many filters there are by explicitly thresholding the
        // singular values of the filters like this:
        // detector = threshold_filter_singular_values(detector,0.3);
        // cout << "remaining filters: "<< num_separable_filters(detector) << endl;
        // That removes filter components with singular values less than 0.1.  The bigger
        // this number the fewer separable filters you will have and the faster the
        // detector will run.  However, a large enough threshold will hurt detection
        // accuracy.  

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------



