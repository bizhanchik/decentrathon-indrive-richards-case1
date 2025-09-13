/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      keyframes: {
        'scanHorizontal': {
          '0%': { transform: 'translateY(-100%)', opacity: '0' },
          '50%': { opacity: '1' },
          '100%': { transform: 'translateY(100vh)', opacity: '0' }
        },
        'scanVertical': {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '50%': { opacity: '1' },
          '100%': { transform: 'translateX(100vw)', opacity: '0' }
        },
        'scanDiagonal1': {
          '0%': { transform: 'translateX(-100%) rotate(45deg)', opacity: '0' },
          '50%': { opacity: '0.6' },
          '100%': { transform: 'translateX(100vw) rotate(45deg)', opacity: '0' }
        },
        'scanDiagonal2': {
          '0%': { transform: 'translateX(100%) rotate(-45deg)', opacity: '0' },
          '50%': { opacity: '0.6' },
          '100%': { transform: 'translateX(-100vw) rotate(-45deg)', opacity: '0' }
        },
        'gridPulse': {
          '0%, 100%': { opacity: '0.1' },
          '50%': { opacity: '0.3' }
        },
        'progress': {
          '0%': { width: '0%' },
          '100%': { width: '100%' }
        },
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        }
      },
      animation: {
        'fade-in': 'fade-in 0.5s ease-out'
      }
    },
  },
  plugins: [],
};
