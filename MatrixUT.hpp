/* 
    Matrix linear algebra library for ML applications (ML: Machine Learning)
    Some errors were corrected, many methods and constructors added

    (c) Lesept june 2022  lesept777@gmail.com

*/


#ifndef __ML_MATRIX_HPP
#define __ML_MATRIX_HPP

#include <Arduino.h>

template <typename T> class MLMatrix {
  private:
    std::vector<std::vector<T> > mat;
    size_t rows;
    size_t cols;

  public:
    MLMatrix() = default;
    template<typename U> explicit MLMatrix(MLMatrix<U> const &, const T = 0);
    template<typename U> explicit MLMatrix(const std::vector<U>&);
    template<typename U> explicit MLMatrix(const std::vector<std::vector<U> >&);
    template<typename U> explicit MLMatrix(const U rhs[], const unsigned);
    explicit MLMatrix(unsigned, unsigned, const T, const T);
    explicit MLMatrix(unsigned, unsigned, const T = 0);
    virtual ~MLMatrix();

    // Access the individual elements: mat(i,j)
    typename std::vector<T>::reference operator()(const unsigned& row, const unsigned& col);
    T const operator()(const unsigned& row, const unsigned& col) const;

    // Operator overloading, for "standard" mathematical matrix operations                                                                                                                                                          
    template<typename U> MLMatrix<T>& operator=(const MLMatrix<U>&);    // copy a matrix
    template<typename U> MLMatrix<T>& operator=(const std::vector<U>&); // copy a vector
    template<typename U> MLMatrix<T>& operator=(const std::vector<std::vector<U> >&);  // copy a vector of vectors
    template<typename U> MLMatrix<T>& fromArray(const U rhs[], const unsigned); // copy an array
    template<typename U> MLMatrix<T>& operator+=(const MLMatrix<U>&);
    template<typename U> MLMatrix<T>& operator-=(const MLMatrix<U>&);
    template<typename U> MLMatrix<T>& operator*=(const MLMatrix<U>&);

    // Matrix comparison
    template<typename U> const bool  operator==(const MLMatrix<U> &) const;
    template<typename U> const bool  operator!=(const MLMatrix<U> &) const;
    template<typename U> MLMatrix<bool> operator< (const MLMatrix<U> &);
    template<typename U> MLMatrix<bool> operator>=(const MLMatrix<U> &);

    // Operations on matrices
    MLMatrix<T> transpose();
    MLMatrix<T> square ();
    template<typename U> MLMatrix<T> Hadamard (const MLMatrix<U>& rhs, bool clip=false);

    // Matrix/scalar operations                                                                                                                                                                                                     
    template<typename U> MLMatrix<T>& operator+=(const U& rhs);
    template<typename U> MLMatrix<T>& operator-=(const U& rhs);
    template<typename U> MLMatrix<T>& operator*=(const U& rhs);
    template<typename U> MLMatrix<T>& operator/=(const U& rhs);

    // Matrix/vector operations                                                                                                                                                                                                     
    template<typename U> std::vector<T>& operator*=(const std::vector<U>& rhs);
    std::vector<T> diag_vec();

    // Vector operations
    template<typename U> auto MdotProd(const MLMatrix<U>& rhs, bool clip=false) -> decltype(std::declval<U>()*std::declval<T>());
    // MLMatrix<T> times(const MLMatrix<T>& rhs, bool clip=false);
    
    // Extract row or col
    MLMatrix<T> row(const uint16_t);
    MLMatrix<T> col(const uint16_t);
    MLMatrix<T> subMatrix(const uint16_t, const uint16_t, const uint16_t, const uint16_t);

    // Pruning functions
    std::vector<T> sortValues(bool = true);
    bool zeroRow(int);
    bool zeroCol(int);
    uint16_t countZeroRow(int);
    uint16_t countZeroCol(int);
    // Remove element
    MLMatrix<T> removeRow(const uint16_t);
    MLMatrix<T> removeCol(const uint16_t); 

    // Norms
    int L0Norm();
    T L1Norm();
    T L2Norm();
    T max() const;
    T min() const;
    float mean() const;
    float stdev(const float) const;
    float meanRow(int);
    // Find min and max values index
    void indexMin(int &, int &);
    void indexMax(int &, int &);
    
    // Display the matrix
    void print();
    void printBool();
    void printSize(); // Only display the size (rows, cols)

    // Access the row and column sizes                                                                                                                                                                                              
    unsigned get_rows() const;
    unsigned get_cols() const;
    void setSize(const int _rows, const int _cols, const T = T(0));

    // Misc 
    MLMatrix<T> applySelf(T (*function)(T));
    MLMatrix<T> apply(T (*function)(T)) ;
    MLMatrix<T> randomChange(const float);
    MLMatrix<T> randomNormal(const float, const float);
    MLMatrix<T> normScale (float, bool &);
    bool normScale2 (float);
    int clipToZero (float);
    int clipMin (float);
    int clipMax (float);
    MLMatrix<T> sgn();
    void setZeroCol(const int);
    void setZeroRow(const int);
    void setCol(const int, const T);
    void setRow(const int, const T);
    void setColMat(const int, const MLMatrix<T>);
    void setRowMat(const int, const MLMatrix<T>);
    MLMatrix<uint8_t>  dropout(const float);

    // DeepShift methods
    MLMatrix<uint8_t> matRound(uint8_t = 10);

};

template<typename U,typename T> auto operator+(
    MLMatrix<U> const &lhs,
    MLMatrix<T> const &rhs) ->
    MLMatrix<decltype(std::declval<U>()+std::declval<T>())>
{
  if ( lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols() ) { // matrices of different sizes
    Serial.printf("Addition error: dimensions do not match (%d, %d)+(%d, %d)", 
      lhs.get_rows(), lhs.get_cols(), rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  MLMatrix<decltype(std::declval<U>()+std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) + rhs(i,j);
  return Res;
}

template<typename U,typename T> auto operator-(
    MLMatrix<U> const &lhs,
    MLMatrix<T> const &rhs) ->
    MLMatrix<decltype(std::declval<U>()-std::declval<T>())>
{
  if ( lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols() ) { // matrices of different sizes
    Serial.printf("Substraction error: dimensions do not match (%d, %d)-(%d, %d)", 
      lhs.get_rows(), lhs.get_cols(), rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  MLMatrix<decltype(std::declval<U>()-std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) - rhs(i,j);
  return Res;
}

template<typename U,typename T> auto operator*(
    MLMatrix<U> const &lhs,
    MLMatrix<T> const &rhs) ->
    MLMatrix<decltype(std::declval<U>()*std::declval<T>())>
{
  if ( lhs.get_cols() != rhs.get_rows() ) { // matrices of different sizes
    Serial.printf("Multiplication error: dimensions do not match (%d, %d)*(%d, %d)", 
      lhs.get_rows(), lhs.get_cols(), rhs.get_rows(), rhs.get_cols());
    while(1);
  }

  MLMatrix<decltype(std::declval<U>()*std::declval<T>())> Res(lhs.get_rows(), rhs.get_cols(), 0);
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < rhs.get_cols(); ++j)
      for (size_t k = 0; k < rhs.get_rows(); ++k)
        Res(i,j) += lhs(i,k) * rhs(k,j);
  return Res;
}

// Matrix/scalar operations
template<typename U,typename T> auto operator+(
    MLMatrix<U> const &lhs,
    T const &rhs) ->
    MLMatrix<decltype(std::declval<U>()+std::declval<T>())>
{
  MLMatrix<decltype(std::declval<U>()+std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) + rhs;
  return Res;
}

template<typename U,typename T> auto operator-(
    MLMatrix<U> const &lhs,
    T const &rhs) ->
    MLMatrix<decltype(std::declval<U>()-std::declval<T>())>
{
  MLMatrix<decltype(std::declval<U>()-std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) - rhs;
  return Res;
}

template<typename U,typename T> auto operator*(
    MLMatrix<U> const &lhs,
    T const &rhs) ->
    MLMatrix<decltype(std::declval<U>()*std::declval<T>())>
{
  MLMatrix<decltype(std::declval<U>()*std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) * rhs;
  return Res;
}

template<typename U,typename T> auto operator/(
    MLMatrix<U> const &lhs,
    T const &rhs) ->
    MLMatrix<decltype(std::declval<U>()/std::declval<T>())>
{
  if (rhs == 0) {
    Serial.println("Error: division by 0");
    return lhs;
  }
  MLMatrix<decltype(std::declval<U>()+std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) / rhs;
  return Res;
}

template<typename U,typename T> auto operator+(
    T const &rhs, MLMatrix<U> const &lhs) ->
    MLMatrix<decltype(std::declval<U>()+std::declval<T>())>
{
  MLMatrix<decltype(std::declval<U>()+std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) + rhs;
  return Res;
}

template<typename U,typename T> auto operator-(
    T const &rhs, MLMatrix<U> const &lhs) ->
    MLMatrix<decltype(std::declval<U>()-std::declval<T>())>
{
  MLMatrix<decltype(std::declval<U>()-std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) - rhs;
  return Res;
}

template<typename U,typename T> auto operator*(
    T const &rhs, MLMatrix<U> const &lhs) ->
    MLMatrix<decltype(std::declval<U>()*std::declval<T>())>
{
  MLMatrix<decltype(std::declval<U>()*std::declval<T>())> Res(lhs.get_rows(), lhs.get_cols());
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res(i,j) = lhs(i,j) * rhs;
  return Res;
}

// Matrix/vector operations
template<typename U,typename T> auto operator*(
    MLMatrix<T> const &lhs, std::vector<U> const &rhs) ->
    std::vector<decltype(std::declval<T>()*std::declval<U>())>
{
  if (lhs.get_cols() != rhs.size()) { 
    Serial.printf("Multiplication error: dimensions do not match (%d, %d)*(%d)", 
      lhs.get_rows(), lhs.get_cols(), rhs.size());
    while(1);
  }

  std::vector<decltype(std::declval<T>()*std::declval<U>())> Res(lhs.get_rows(), 0);
  for (size_t i = 0; i < lhs.get_rows(); ++i)
    for (size_t j = 0; j < lhs.get_cols(); ++j)
      Res[i] += lhs(i,j) * rhs[j];
  return Res;
}


#include "MatrixUT.tpp"
#endif
