# Fruit classification project with DNN algorithm

## Thuật toán K-Nearest Neighbor
###	Thuật toán KNN
####	Định nghĩa
K – nearest neighbor (KNN) là một trong những thuật toán học có giám sát đơn giản nhất trong Machine Learning. Ý tưởng của KNN là tìm ra output của dữ liệu dựa trên thông tin của những dữ liệu training gần nó nhất.
Kỹ thuật phân lớp dựa trên láng giềng gần nhất được sử dụng rộng rãi trong các hệ thống nhận dạng mẫu, nhận dạng đối tượng, nhận dạng sự kiện, phân loại dữ liệu văn bản… Khái niệm “láng giềng” dùng để chỉ các đối tượng có khoảng cách hoặc độ tương đồng “gần” với đối tượng x. Từ đây, ta cần phải định nghĩa một độ đo khoảng cách hoặc độ đo sự khác biệt giữa các đối tượng.
####	Quy trình làm việc của thuật toán KNN
*	Bước 1: Xác định tham số K = số láng giềng gần nhất
*	Bước 2: Tính khoảng cách đối tượng cần phần lớp với tất cả các đối tượng trong training data.
*	Bước 3: Sắp xếp khoảng cách theo thứ tự tăng dần và xác định K láng giềng gần nhất với đối tượng cần phân lớp
*	Bước 4: Lấy tất cả các lớp của K láng giềng gần nhất.
*	Bước 5: Dựa vào phần lớn lớp của K để xác định lớp cho đối tượng cần phân lớp.

#### Ví dụ về KNN nhiễu

![image](https://github.com/ThanhTungPh2/Fruit-classification/assets/78742865/57fc412c-301e-4653-b2d3-37c723a45bcf)

Hình trên là bài toán phân lớp với ba lớp: đỏ, lam, lục. Mỗi điểm dữ liệu mới sẽ được gán nhãn theo màu của điểm đó mà nó thuộc về. Trong hình này, chú ý vùng khoanh tròn. Ta nhận thấy rằng điểm màu lục nằm giữa hai vùng lớn với nhiều dữ liệu đỏ và lam, điểm này rất có thể là nhiễu dẫn đến việc dữ liệu test nếu rơi vào vùng này sẽ có nhiều khả năng cho kết quả sai lệch.